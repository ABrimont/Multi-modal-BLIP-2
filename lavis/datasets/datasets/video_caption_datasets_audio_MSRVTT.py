"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
import torchaudio
import soundfile as sf
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset

# ================================================================
#   Utility function: pad / truncate audio + masks (Cleaned)
# ================================================================
def pad_or_truncate_audio(audio, valid, target_length=320250, frame_hop=320, max_frames=1000):
    """
    Cuts or pads an audio to a fixed length and creates BEATs / LLM masks.
    audio: [1, T]
    valid: 1 if audio is valid, 0 otherwise
    """
    audio = audio[:, :target_length]
    real_len = audio.shape[1] 
    pad_len = target_length - real_len
    
    if pad_len > 0:
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    mask_BEATs = torch.zeros(1, target_length, dtype=torch.bool)
    if pad_len > 0:
        mask_BEATs[:, -pad_len:] = True

    mask_LLM = torch.zeros(1, max_frames, dtype=torch.long)
    if valid == 1:
        n_frames = min(real_len // frame_hop, max_frames)  
        mask_LLM[:, :n_frames] = 1

    return audio, mask_BEATs, mask_LLM


# ================================================================
#   MSRVTT Dataset — Training
# ================================================================
class VideoCaptionDatasetAudioMSRVTT(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.audio_root = self.vis_root.replace("raw_videos", "raw_audios")

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        caption = ann.get("caption", "")
        # Security fallback to index if "image_id" is missing
        image_id = ann.get("image_id", str(index))

        # ------------------------
        #   VIDEO
        # ------------------------
        try:
            video_path = os.path.join(self.vis_root, vname)
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                video_path = os.path.join(self.vis_root, vname + ".mkv")

            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                video_path = os.path.join(self.vis_root, vname + ".mp4")
                
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Missing or empty video file: {video_path}")

            video = self.vis_processor(video_path)
            if video is None or not torch.is_tensor(video):
                raise ValueError(f"vis_processor returned None or invalid tensor for {video_path}")

            valid_video = 1

        except Exception as e:
            print(f"[WARN] Failed to load video {vname}: {e}")
            # MSRVTT specific dummy video -> black tensor of 8 frames
            video = torch.zeros(3, 16, 224, 224)  
            valid_video = 0

        # ------------------------
        #   AUDIO
        # ------------------------
        try:
            # Clean extraction of the base name, without depending on [:-4]
            base_vname = os.path.splitext(os.path.basename(vname))[0]
            audio_file = os.path.join(self.audio_root, base_vname + ".wav")

            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform is None or len(waveform) == 0:
                raise ValueError(f"Empty or None waveform in {audio_file}")

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            waveform = torch.tensor(waveform).unsqueeze(0)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            valid_audio = 1

        except Exception as e:
            # print(f"[WARN] Failed to load audio {audio_file}: {e}")
            # MSRVTT specific fallback -> absolute zeros
            waveform = torch.zeros(1, 320250)
            valid_audio = 0

        # ------------------------
        #   PADDING & MASKS
        # ------------------------
        audio, mask_BEATs, mask_LLM = pad_or_truncate_audio(
            waveform, 
            valid_audio, 
            target_length=320250, 
            max_frames=1000
        )

        return {
            "video": video,
            "image": video, # Ensures compatibility with other modules
            "audio": audio,
            "audio_mask": mask_BEATs,
            "audio_mask_LLM": mask_LLM,
            "text_input": caption,
            "image_id": image_id, 
            "valid_video": valid_video,
            "valid_audio": valid_audio,
        }


# ================================================================
#   MSRVTT Dataset — Eval
# ================================================================
class VideoCaptionEvalDatasetAudioMSRVTT(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.audio_root = self.vis_root.replace("raw_videos", "raw_audios")

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        caption = ann.get("caption", "")
        image_id = ann.get("image_id", str(index))

        # ------------------------
        #   VIDEO
        # ------------------------
        try:
            video_path = os.path.join(self.vis_root, vname)
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                video_path = os.path.join(self.vis_root, vname + ".mkv")

            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                video_path = os.path.join(self.vis_root, vname + ".mp4")
                
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Missing or empty video file: {video_path}")

            video = self.vis_processor(video_path)
            if video is None or not torch.is_tensor(video):
                raise ValueError(f"vis_processor returned None or invalid tensor for {video_path}")

            valid_video = 1

        except Exception as e:
            print(f"[WARN] Failed to load eval video {vname}: {e}")
            video = torch.zeros(3, 16, 224, 224)  
            valid_video = 0

        # ------------------------
        #   AUDIO
        # ------------------------
        try:
            base_vname = os.path.splitext(os.path.basename(vname))[0]
            audio_file = os.path.join(self.audio_root, base_vname + ".wav")

            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform is None or len(waveform) == 0:
                raise ValueError(f"Empty or None waveform in {audio_file}")

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            waveform = torch.tensor(waveform).unsqueeze(0)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            valid_audio = 1

        except Exception as e:
            waveform = torch.zeros(1, 320250)
            valid_audio = 0

        # ------------------------
        #   PADDING & MASKS
        # ------------------------
        audio, mask_BEATs, mask_LLM = pad_or_truncate_audio(
            waveform, 
            valid_audio, 
            target_length=320250, 
            max_frames=1000
        )

        return {
            "video": video,
            "image": video,
            "audio": audio,
            "audio_mask": mask_BEATs,
            "audio_mask_LLM": mask_LLM,
            "text_input": caption,
            "image_id": image_id,
            "valid_video": valid_video,
            "valid_audio": valid_audio,
        }