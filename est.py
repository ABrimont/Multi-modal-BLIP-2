# import torch

# # Load the checkpoint
# ckpt = torch.load(
#     "/home/abrimont/partage/mllm-video-captioner/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth",
#     map_location="cpu"
# )

# # Extract state_dict
# if "state_dict" in ckpt:
#     state_dict = ckpt["state_dict"]
# elif "model" in ckpt:
#     state_dict = ckpt["model"]
# else:
#     state_dict = ckpt

# qformer_keys = [k for k in state_dict.keys() if k.startswith("Qformer")]
# print("\nNombre de clés Qformer:", len(qformer_keys))
# print("\nExemples:")
# for k in qformer_keys:
#     print(" ", k)

# # Compter tous les paramètres
# total_params = sum(t.numel() for t in state_dict.values())
# print(f"\nTotal parameters: {total_params:,}")

# # Compter spécifiquement les poids liés aux cross-attention
# cross_params = 0
# cross_keys = []
# for name, tensor in state_dict.items():
#     if "crossattention" in name.lower():  # dépend du naming exact
#         cross_params += tensor.numel()
#         cross_keys.append(name)

# print(f"Cross-attention parameters: {cross_params:,}")

# # Montrer un échantillon de clés trouvées
# print("\nExample cross-attention parameter keys:")
# for k in cross_keys[:20]:
#     print(" ", k)

# # Taille du Q-Former si on double les cross-attention
# qformer_params = sum(t.numel() for n, t in state_dict.items() if n.startswith("Qformer"))
# print(f"\nCurrent Q-Former parameters: {qformer_params:,}")
# print(f"Estimated if cross-attention doubled: {qformer_params + cross_params:,}")


# # --- Compter et extraire Self-Attention (SA) ---
# sa_params = 0
# sa_keys = []
# for name, tensor in state_dict.items():
#     if "attention.self" in name.lower():  # self-attention Q/K/V
#         sa_params += tensor.numel()
#         sa_keys.append(name)

# print(f"\nSelf-Attention parameters: {sa_params:,}")
# print("Example SA parameter keys:")
# for k in sa_keys[:20]:
#     print(" ", k)

# # --- Compter et extraire Feed-Forward (FFN) ---
# ffn_params = 0
# ffn_keys = []
# for name, tensor in state_dict.items():
#     if "intermediate" in name.lower() or "output.dense" in name.lower():
#         ffn_params += tensor.numel()
#         ffn_keys.append(name)

# print(f"\nFeed-Forward parameters: {ffn_params:,}")
# print("Example FFN parameter keys:")
# for k in ffn_keys[:20]:
#     print(" ", k)


# import torch

# ckpt_path = "/home/abrimont/partage/mllm-video-captioner/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xl.pth"
# import pickle


# checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
# qformer_keys = [k for k in checkpoint['model'].keys() if k.startswith("Qformer")]
# print(len(qformer_keys), "Qformer keys trouvées")
# print(qformer_keys[:100])  # affiche les 50 premières pour inspection

# import torch
# from unilm.beats.BEATs import BEATs, BEATsConfig

# # load the pre-trained checkpoints
# checkpoint = torch.load(
#     "/home/abrimont/partage/mllm-video-captioner/BEATs_iter3_plus_AS2M.pt",
#     map_location="cpu"
# )

# cfg = BEATsConfig(checkpoint["cfg"])
# BEATs_model = BEATs(cfg)
# BEATs_model.load_state_dict(checkpoint["model"])
# BEATs_model.eval()

# # extract an audio representation
# audio_input_16khz = torch.randn(1, 16000)  # 1 sec de bruit blanc
# padding_mask = torch.zeros(1, 16000).bool()

# representation = BEATs_model.extract_features(
#     audio_input_16khz,
#     padding_mask=padding_mask
# )[0]

# print("Representation shape:", representation.shape)
# print(representation)

# import torch
# from unilm.beats.BEATs import BEATs, BEATsConfig  # adapte ton chemin d'import

# # Charger checkpoint
# checkpoint = torch.load("BEATs_iter3_plus_AS2M(2).pt", map_location="cpu")

# # Config et modèle
# cfg = BEATsConfig(checkpoint['cfg'])
# model = BEATs(cfg)
# model.load_state_dict(checkpoint['model'])

# # Infos structurelles
# print("Hidden size:", cfg.encoder_embed_dim)
# print("Nb de couches:", cfg.encoder_layers)
# print("Nb de têtes d'attention:", cfg.encoder_attention_heads)

# # Nombre total de paramètres
# n_params = sum(p.numel() for p in model.parameters())
# n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Nombre total de paramètres: {n_params/1e6:.2f} M")
# print(f"Nombre de paramètres entraînables: {n_params_trainable/1e6:.2f} M")
# # import torch

# # Load checkpoint
# ckpt_path = "/home/abrimont/partage/VALOR/output/VALOR_large/cap-valor32k-lr2e-5-bs64/ckpt/best_cap%tva%tv--msrvtt_cap_tv.pt"
# checkpoint = torch.load(ckpt_path, map_location="cpu")

# # If it's a full state_dict
# state_dict = checkpoint.get("model", checkpoint)

# # Count parameters
# n_params = sum(p.numel() for p in state_dict.values())
# print(f"Total parameters in checkpoint: {n_params:,}")



# from transformers import pipeline
# import torch, time

# question = "Qu'est ce que Télécom SudParis ? "

# # === 🚀 FLAN-T5-XL (2022) ===
# t0 = time.time()
# flan = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-xl",
#     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )
# flan_answer = flan(question, max_length=100, temperature=0.7)[0]["generated_text"]
# t_flan = time.time() - t0

# # === 🦙 LLaMA-1-7B (2023) ===
# t0 = time.time()
# llama = pipeline(
#     "text-generation",
#     model="huggyllama/llama-7b",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# llama_prompt = f"Answer in English: {question}"
# llama_answer = llama(llama_prompt, max_new_tokens=512, temperature=0.6, do_sample=True)[0]["generated_text"]
# t_llama = time.time() - t0

# # === 🔥 Mistral-7B-Instruct-v0.3 (2024) ===
# t0 = time.time()
# mistral = pipeline(
#     "text-generation",
#     model="mistralai/Mistral-7B-Instruct-v0.3",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# mistral_answer = mistral(question, max_new_tokens=512, temperature=0.4, do_sample=True)[0]["generated_text"]
# t_mistral = time.time() - t0

# # === 🧾 Résumé ===
# print("\n🧩 COMPARAISON DE MODÈLES\n")
# print(f"❓ Question : {question}\n")
# print(f"--- FLAN-T5-XL (2022) --- [{t_flan:.1f}s]\n{flan_answer.strip()}\n")
# print(f"--- LLaMA-1 7B (2023) --- [{t_llama:.1f}s]\n{llama_answer.strip()}\n")
# print(f"--- Mistral 7B Instruct (2024) --- [{t_mistral:.1f}s]\n{mistral_answer.strip()}\n")
# from transformers import T5ForConditionalGeneration

# t5_model_name = "google/flan-t5-xl"
# model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# print(f"\n✅ Modèle T5 chargé : {t5_model_name}")
# print(f"Nombre total de paramètres : {sum(p.numel() for p in model.parameters()):,}\n")

# print("🔑 --- Exemples de clés de poids ---")
# for i, (name, param) in enumerate(model.state_dict().items()):
#     print(f"{i:3d}: {name:70s}  {tuple(param.shape)}")
#     if i >= 40:
#         print("...")
#         break
import json

base_path = "/home/abrimont/partage/AudioCaps"

with open(f"{base_path}/train.json") as f:
    train = json.load(f)
with open(f"{base_path}/val_unique.json") as f:
    val = json.load(f)
with open(f"{base_path}/test_unique.json") as f:
    test = json.load(f)

def get_ids(data):
    return {item["image_id"] for item in data}

train_ids = get_ids(train)
val_ids = get_ids(val)
test_ids = get_ids(test)

# Vérifier intersections
inter_train_val = train_ids & val_ids
inter_train_test = train_ids & test_ids
inter_val_test = val_ids & test_ids

print(f"Train ∩ Val : {len(inter_train_val)}")
print(f"Train ∩ Test : {len(inter_train_test)}")
print(f"Val ∩ Test : {len(inter_val_test)}")

print(f"Total uniques : {len(train_ids | val_ids | test_ids)}")
print(f"Somme des trois : {len(train_ids) + len(val_ids) + len(test_ids)}")
