import os
import torchaudio
import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset
import soundfile as sf


def pad_or_truncate_audio(audio, valid, target_length=160125, frame_hop=320, max_frames=496):
    """
    audio: [1, T] waveform
    valid: 1 si audio valide, 0 sinon
    """
    # Tronquer / padder waveform
    audio = audio[:, :target_length]
    pad_len = target_length - audio.shape[1]
    padded = torch.nn.functional.pad(audio, (0, pad_len))  # [1, target_length]

    # Masque BEATs = même taille que waveform (True=padding, False=valide)
    mask_BEATs = torch.zeros(1, target_length, dtype=torch.bool)
    if pad_len > 0:
        mask_BEATs[:, -pad_len:] = True

    # Masque LLM: 1 = valide, 0 = padding
    mask_LLM = torch.zeros(1, max_frames, dtype=torch.long)
    if valid == 1:
        n_frames = padded.shape[1] // frame_hop
        n_frames = min(n_frames, max_frames)
        mask_LLM[:, :n_frames] = 1
    # print(
    #         f"[DEBUG] audio_len={padded.shape[1]}, "
    #         f"mask_BEATs padding={mask_BEATs.sum().item()}, "
    #         f"mask_BEATs valids={(~mask_BEATs).sum().item()}, "
    #         f"mask_LLM valids={mask_LLM.sum().item()}"
    #     )


    return padded, mask_BEATs, mask_LLM




class VideoCaptionDatasetAudioVATEX(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.audio_root = "/home/abrimont/partage/VALOR/datasets/vatex/audio_22050hz/"
        self.vis_root = "/home/abrimont/partage/VALOR/datasets/vatex/raw_videos/"

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")

        # --- Vidéo ---
        try:
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Video file missing or empty: {video_path}")
            video = self.vis_processor(video_path)
            valid_video = 1
        except Exception as e:
            print(f"[WARN] Failed to load video {video_path}: {e}")
            video = torch.zeros(3, 16, 224, 224)
            valid_video = 0

        caption = ann.get("caption", "")

        # --- Audio ---
        try:
            audio_file = os.path.join(
                self.audio_root, os.path.splitext(os.path.basename(vname))[0]
            ) + ".wav"
            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # mono
            waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            valid_audio = 1
        except Exception as e:
            print(f"[WARN] Failed to load audio {audio_file}: {e}")
            waveform = torch.randn(1, 160125) * 1e-3  # bruit faible
            mask_BEATs = torch.zeros(1,160125)
            valid_audio = 0

        audio, mask_BEATs, mask_LLM = pad_or_truncate_audio(waveform, valid_audio)

        return {
            "video": video,
            "audio": audio,
            "audio_mask": mask_BEATs,       # [1, n_frames]
            "audio_mask_LLM": mask_LLM,     # [1, max_frames]
            "text_input": caption,
            "image_id": ann["image_id"],
            "valid_video": valid_video,
            "valid_audio": valid_audio,
        }


class VideoCaptionEvalDatasetAudioVATEX(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.audio_root = "/home/abrimont/partage/VALOR/datasets/vatex/audio_22050hz"
        self.vis_root = "/home/abrimont/partage/VALOR/datasets/vatex/raw_videos"

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname + ".mp4")

        # --- Vidéo ---
        try:
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Video file missing or empty: {video_path}")
            video = self.vis_processor(video_path)
            valid_video = 1
        except Exception as e:
            print(f"[WARN] Failed to load video {video_path}: {e}")
            video = torch.zeros(3, 16, 224, 224)
            valid_video = 0

        caption = ann.get("caption", "")

        # --- Audio ---
        try:
            audio_file = os.path.join(
                self.audio_root, os.path.splitext(os.path.basename(vname))[0]
            ) + ".wav"
            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.tensor(waveform).unsqueeze(0)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            valid_audio = 1
        except Exception as e:
            print(f"[WARN] Failed to load audio {audio_file}: {e}")
            waveform = torch.randn(1, 160125) * 1e-3
            mask_BEATs = torch.zeros(1,160125)
            valid_audio = 0

        audio, mask_BEATs, mask_LLM = pad_or_truncate_audio(waveform, valid_audio)

        return {
            "video": video,
            "audio": audio,
            "audio_mask": mask_BEATs,
            "audio_mask_LLM": mask_LLM,
            "text_input": caption,
            "image_id": ann["image_id"],
            "valid_video": valid_video,
            "valid_audio": valid_audio,
        }
