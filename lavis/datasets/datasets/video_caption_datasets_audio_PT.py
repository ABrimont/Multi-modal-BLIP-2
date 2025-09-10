"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
import torchaudio
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset
import soundfile as sf
import subprocess, torch, numpy as np


class VideoCaptionDatasetAudio_PT(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos
        ann_paths (list): List of annotation file paths
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    @staticmethod
    def pad_or_truncate_audio(audio, valid, target_length=480_000):
        audio = audio[:, :target_length]
        pad_len = target_length - audio.shape[1]
        padded = torch.nn.functional.pad(audio, (0, pad_len))
        mask_BEATs = torch.cat([
            torch.zeros(1, audio.shape[1]),
            torch.ones(1, pad_len)
        ], dim=1).bool()

        if valid == 0:
            mask_LLM = torch.zeros(1, 1496)
        else:
            n_valid = min(1496, audio.shape[1] // 320)
            mask_LLM = torch.cat([
                torch.ones(1, n_valid),
                torch.zeros(1, 1496 - n_valid)
            ], dim=1)

        return padded, mask_BEATs, mask_LLM


    @staticmethod
    def ffmpeg_load_audio(video_path, target_sr=16000):
        cmd = [
            "ffmpeg", "-i", video_path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(target_sr), "-ac", "1", "pipe:1"
        ]
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        audio = np.frombuffer(out.stdout, np.float32)
        return torch.tensor(audio).unsqueeze(0), target_sr


    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        # --- Chargement vidéo ---
        try:
            video = self.vis_processor(video_path)  # peut lever DECORDError
            video_valid = 1
        except Exception as e:
            print(f"[WARN] Échec du chargement vidéo {video_path}: {e}")
            # Placeholder : tensor noir, même shape que prévu par ton modèle
            # À adapter selon ce que vis_processor retourne normalement
            video = torch.zeros((3, 8, 224, 224))
            video_valid = 0

        caption = ann['caption']

        # --- Extraction audio ---
        try:
            waveform, sr = self.ffmpeg_load_audio(video_path)
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            audio_valid = 1
        except Exception as e:
            print(f"[WARN] Échec de l'extraction audio {video_path}: {e}")
            waveform = torch.zeros(1, 480_000)
            audio_valid = 0

        audio, mask_BEATs, mask_LLM = self.pad_or_truncate_audio(waveform, audio_valid)

        return {
            "video": video,
            "video_valid": video_valid,
            "audio": audio,
            "audio_mask": mask_BEATs,
            "audio_mask_LLM": mask_LLM,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }



class VideoCaptionEvalDatasetAudio_PT(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.audio_root = "/home/abrimont/partage/VALOR/datasets/msrvtt/audio_22050hz"

    @staticmethod
    def pad_or_truncate_audio(audio, valid, target_length=480000):
        audio = audio[:, :target_length]
        pad_len = target_length - audio.shape[1]
        padded = torch.nn.functional.pad(audio, (0, pad_len))
        mask_BEATs = torch.cat([torch.zeros(1, audio.shape[1]), torch.ones(1, pad_len)], dim=1)
        if valid == 0:
            mask_LLM = torch.zeros(1, 1496)
        else:
            n_valid = min(1496, audio.shape[1] // 320)
            mask_LLM = torch.cat([torch.ones(1, n_valid), torch.zeros(1, 1496 - n_valid)], dim=1)
        return padded, mask_BEATs, mask_LLM

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        try:
            audio_file = os.path.join(self.audio_root, vname[:-4]) + ".wav"
            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            valid = 1
        except Exception as e:
            # print(f"[WARN] Failed to load audio {audio_file}: {e}")
            waveform = torch.zeros(1, 480_000)
            valid = 0

        audio, mask_BEATs, mask_LLM = self.pad_or_truncate_audio(waveform, valid)

        return {
            "video": video,
            "audio":audio,
            "audio_mask":mask_BEATs,
            "audio_mask_LLM":mask_LLM,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
    
    
