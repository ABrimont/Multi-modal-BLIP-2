"""
Copyright (c) 2025
SPDX-License-Identifier: BSD-3-Clause
"""

import itertools
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.tasks.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from lavis.tasks.pycocoevalcap.cider.cider import Cider

from .unilm.beats.BEATs import BEATs, BEATsConfig
from .blip2_qf_dual_path import init_Qformer_dual


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def tokenize(refs, cands):
    """Utility for CIDEr score computation (SCST)."""
    tokenizer = PTBTokenizer()
    refs = {i: [{"caption": r} for r in rr] for i, rr in enumerate(refs)}
    cands = {i: [{"caption": c}] for i, c in enumerate(cands)}
    return tokenizer.tokenize(refs), tokenizer.tokenize(cands)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
@registry.register_model("blip2_smollm_multimodal")
class Blip2SmolLM_multimodal(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_smollm3": "configs/models/blip2/blip2_pretrain_smollm3.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        freeze_vit=True,
        num_query_token=48,
        llm_name="Qwen/Qwen2.5-3B",
        prompt="",
        max_txt_len=128,
        scst=False,
        beam_size=5,
        lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Audio Encoder (BEATs)
        # ------------------------------------------------------------------
        checkpoint_path = (
            "/home/abrimont/partage/mllm-video-captioner/"
            "BEATs_iter3_plus_AS2M(2).pt"
        )
        print(f"Loading Audio Encoder from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        self.audio_encoder = BEATs(cfg)
        self.audio_encoder.load_state_dict(checkpoint["model"])
        self.audio_encoder.eval().to(torch.float32)

        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        self.ln_audio = nn.LayerNorm(768)

        # ------------------------------------------------------------------
        # 2. Vision Encoder (EVA-CLIP)
        # ------------------------------------------------------------------
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, 0, False, "fp16"
        )

        if freeze_vit:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        # ------------------------------------------------------------------
        # 3. Q-Former (Dual Path)
        # ------------------------------------------------------------------
        self.Qformer, self.query_tokens = init_Qformer_dual(
            num_query_token,
            vision_width=1408,
            cross_attention_freq=2,
        )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None

        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # ------------------------------------------------------------------
        # 4. LLM & Tokenizer (Qwen 2.5)
        # ------------------------------------------------------------------
        print(f"Loading LLM: {llm_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            use_fast=False,
            trust_remote_code=True,
            padding_side="right",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if torch.cuda.is_bf16_supported():
            self.train_dtype = torch.bfloat16
            print("Training precision: bfloat16")
        else:
            self.train_dtype = torch.float16
            print("Training precision: float16")

        device_arg = {"device_map": "auto"}
        if torch.cuda.is_available():
            device_arg = {"device_map": {"": torch.cuda.current_device()}}

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=self.train_dtype,
            trust_remote_code=True,
            **device_arg,
        )

        for p in self.llm.parameters():
            p.requires_grad = False

        if lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
            )
            self.llm = get_peft_model(self.llm, peft_cfg)
            self.llm.print_trainable_parameters()

        # ------------------------------------------------------------------
        # 5. Projection Layers
        # ------------------------------------------------------------------
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size,
            self.llm.config.hidden_size,
        )
        self.llm_proj_aud = nn.Linear(
            self.Qformer.config.hidden_size,
            self.llm.config.hidden_size,
        )

        self.prompt = prompt
        self.max_txt_len = max_txt_len
        self.scst = scst
        self.beam_size = beam_size

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, samples):
        audio = samples["audio"].float()
        audio_mask = samples["audio_mask"]
        audio_mask_llm = samples["audio_mask_LLM"]

        with self.maybe_autocast(dtype=torch.float32):
            audio_feat = self.audio_encoder.extract_features(
                audio.squeeze(1),
                padding_mask=audio_mask.squeeze(1).bool(),
            )[0]

        audio_feat = self.ln_audio(audio_feat)

        image = samples["video"]
        B, C, T, H, W = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with self.maybe_autocast():
            vis_feat = self.ln_vision(self.visual_encoder(image))
            vis_feat = vis_feat.reshape(B, -1, vis_feat.shape[-1])

        vis_atts = torch.ones(vis_feat.size()[:-1], device=image.device)

        q_out = self.Qformer.bert(
            query_embeds=self.query_tokens,
            encoder_hidden_states_vis=vis_feat,
            encoder_attention_mask_vis=vis_atts,
            encoder_hidden_states_aud=audio_feat,
            encoder_attention_mask_aud=audio_mask_llm,
            return_dict=True,
        )

        vis_tokens = self.llm_proj(q_out.last_hidden_state[:, :32])
        aud_tokens = self.llm_proj_aud(q_out.last_hidden_state[:, 32:])
        prefix_embeds = torch.cat([aud_tokens, vis_tokens], dim=1)
        prefix_embeds = prefix_embeds.to(self.train_dtype)

        # --------------------------------------------------------------
        # Standard Causal Training
        # --------------------------------------------------------------
        if not self.scst:
            text = [t + self.tokenizer.eos_token for t in samples["text_input"]]

            with self.maybe_autocast(dtype=self.train_dtype):
                tokens = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(prefix_embeds.device)

                text_embeds = self.llm.get_input_embeddings()(tokens.input_ids)
                inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

                prefix_labels = torch.full(
                    (B, prefix_embeds.size(1)),
                    -100,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

                text_labels = tokens.input_ids.clone()
                text_labels.masked_fill_(tokens.attention_mask == 0, -100)

                labels = torch.cat([prefix_labels, text_labels], dim=1)

                outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                )

            return {"loss": outputs.loss}

        # --------------------------------------------------------------
        # SCST (CIDEr)
        # --------------------------------------------------------------
        with self.maybe_autocast(dtype=self.train_dtype):
            outputs = self.llm.generate(
                inputs_embeds=prefix_embeds,
                num_beams=self.beam_size,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            transition_scores = self.llm.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=False,
            )

            seq_lens = (transition_scores < 0).sum(dim=1).clamp(min=1)
            log_probs = transition_scores.sum(dim=1) / seq_lens

        caps_gen = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )

        caps_gt = list(
            itertools.chain(
                *([c] * self.beam_size for c in samples["text_input"])
            )
        )

        caps_gen, caps_gt = tokenize(caps_gt, caps_gen)
        reward = Cider().compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(log_probs.device)
        reward = reward.view(B, self.beam_size).mean(dim=1)

        baseline = reward.mean()
        loss = -((reward - baseline) * log_probs).mean()

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        **kwargs,
    ):
        audio = samples["audio"].float()
        audio_mask = samples["audio_mask"]
        audio_mask_llm = samples["audio_mask_LLM"]

        with self.maybe_autocast(dtype=torch.float32):
            audio_feat = self.audio_encoder.extract_features(
                audio.squeeze(1),
                padding_mask=audio_mask.squeeze(1).bool(),
            )[0]

        audio_feat = self.ln_audio(audio_feat)

        image = samples["video"]
        B, C, T, H, W = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with self.maybe_autocast():
            vis_feat = self.ln_vision(self.visual_encoder(image))
            vis_feat = vis_feat.reshape(B, -1, vis_feat.shape[-1])

        vis_atts = torch.ones(vis_feat.size()[:-1], device=image.device)

        q_out = self.Qformer.bert(
            query_embeds=self.query_tokens,
            encoder_hidden_states_vis=vis_feat,
            encoder_attention_mask_vis=vis_atts,
            encoder_hidden_states_aud=audio_feat,
            encoder_attention_mask_aud=audio_mask_llm,
            return_dict=True,
        )

        vis_tokens = self.llm_proj(q_out.last_hidden_state[:, :32])
        aud_tokens = self.llm_proj_aud(q_out.last_hidden_state[:, 32:])
        prefix_embeds = torch.cat([aud_tokens, vis_tokens], dim=1)
        prefix_embeds = prefix_embeds.to(self.train_dtype)

        gen_kwargs = dict(
            inputs_embeds=prefix_embeds,
            max_new_tokens=50,
            min_length=5,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if use_nucleus_sampling:
            gen_kwargs.update(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                num_beams=1,
            )
        else:
            gen_kwargs["do_sample"] = False

        with self.maybe_autocast(dtype=self.train_dtype):
            outputs = self.llm.generate(**gen_kwargs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        model = cls(
            vit_model=cfg.get("vit_model", "eva_clip_g"),
            img_size=cfg.get("image_size", 224),
            llm_name=cfg.get("llm_model", "Qwen/Qwen2.5-3B"),
            prompt=cfg.get("prompt", ""),
            max_txt_len=cfg.get("max_txt_len", 128),
            scst=cfg.get("scst", False),
            beam_size=cfg.get("beam_size", 5),
            lora=cfg.get("lora", False),
            lora_r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
        )

        model.load_checkpoint_from_config_multimodal(cfg)
        return model
