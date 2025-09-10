"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torchaudio
import torch
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.caption_datasets import CaptionDataset
import soundfile as sf

class VideoCaptionDatasetAudio(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # self.audio_root = "/home/abrimont/partage/VALOR/datasets/vatex/audio_22050hz"
        
        self.audio_root = "/home/abrimont/partage/VALOR/datasets/msrvtt/audio_22050hz"
        self.vis_root= "/home/abrimont/partage/VALOR/datasets/msrvtt/raw_videos"

    @staticmethod
    def pad_or_truncate_audio(audio, valid, target_length=480000):
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
            mask_LLM = torch.cat([torch.ones(1, n_valid), torch.zeros(1, 1496 - n_valid)], dim=1)
        return padded, mask_BEATs, mask_LLM

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)
        # --- Charger la vidéo en mode safe ---
        try:
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Video file missing or empty: {video_path}")
            video = self.vis_processor(video_path)
            valid_video = 1
        except Exception as e:
            print(f"[WARN] Failed to load video {video_path}: {e}")
            # Dummy vidéo -> par ex. tenseur noir 224x224
            video = torch.zeros(3, 8, 224, 224)  
            valid_video = 0
        caption = ann.get("caption", "")

        # --- Charger l’audio en mode safe ---
        try:
            audio_file = os.path.join(self.audio_root, vname[:-4]) + ".wav"
            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            valid_audio = 1
        except Exception as e:
            # print(f"[WARN] Failed to load audio {audio_file}: {e}")
            waveform = torch.zeros(1, 480_000)
            valid_audio = 0

        audio, mask_BEATs, mask_LLM = self.pad_or_truncate_audio(waveform, valid_audio)

        return {
            "video": video,
            "audio": audio,
            "audio_mask": mask_BEATs,
            "audio_mask_LLM": mask_LLM,
            "text_input": caption,
            "image_id": ann["image_id"],  # ✅ direct string ID (filename)
            "valid_video": valid_video,
            "valid_audio": valid_audio,
        }




class VideoCaptionEvalDatasetAudio(BaseDataset):
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

        # --- Charger la vidéo en mode safe ---
        try:
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Video file missing or empty: {video_path}")
            video = self.vis_processor(video_path)
            valid_video = 1
        except Exception as e:
            print(f"[WARN] Failed to load video {video_path}: {e}")
            # Dummy vidéo -> par ex. tenseur noir 224x224
            video = torch.zeros(3, 8, 224, 224)  
            valid_video = 0

        # print(video_path)

        # print(video.shape)
        caption = ann.get("caption", "")

        # --- Charger l’audio en mode safe ---
        try:
            audio_file = os.path.join(self.audio_root, vname[:-4]) + ".wav"
            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            valid_audio = 1
        except Exception as e:
            # print(f"[WARN] Failed to load audio {audio_file}: {e}")
            waveform = torch.zeros(1, 480_000)
            valid_audio = 0

        audio, mask_BEATs, mask_LLM = self.pad_or_truncate_audio(waveform, valid_audio)

        return {
            "video": video,
            "audio": audio,
            "audio_mask": mask_BEATs,
            "audio_mask_LLM": mask_LLM,
            "text_input": caption,
            "image_id": ann["image_id"],  # ✅ direct string ID (filename)
            "valid_video": valid_video,
            "valid_audio": valid_audio,
        }
