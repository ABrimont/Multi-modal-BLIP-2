import os
import torch
import torchaudio
import soundfile as sf
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset


# ================================================================
#   Fonction utilitaire : pad / truncate audio + masques
# ================================================================
def pad_or_truncate_audio(audio, valid, target_length=160125, frame_hop=320, max_frames=496):
    """
    Coupe ou pad un audio à une longueur fixe et crée les masques BEATs / LLM.
    audio: [1, T]
    valid: 1 si audio valide, 0 sinon
    """
    # Tronquer / padder
    audio = audio[:, :target_length]
    pad_len = target_length - audio.shape[1]
    if pad_len > 0:
        audio = torch.nn.functional.pad(audio, (0, pad_len))  # [1, target_length]

    # Masque BEATs (True = padding)
    mask_BEATs = torch.zeros(1, target_length, dtype=torch.bool)
    if pad_len > 0:
        mask_BEATs[:, -pad_len:] = True

    # Masque LLM (1 = frame valide)
    mask_LLM = torch.zeros(1, max_frames, dtype=torch.long)
    if valid == 1:
        n_frames = min(audio.shape[1] // frame_hop, max_frames)
        mask_LLM[:, :n_frames] = 1

    return audio, mask_BEATs, mask_LLM


# ================================================================
#   Dataset AudioCaps — Entraînement
# ================================================================
class VideoCaptionDatasetAudioCaps(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.audio_root = "/home/abrimont/partage/AudioCaps/audiocaps_raw_audio/"
        self.vis_root = "/home/abrimont/partage/AudioCaps/audiocaps_raw_video/"

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        caption = ann.get("caption", "")
        image_id = ann.get("image_id", index)

        # ------------------------
        #   VIDÉO
        # ------------------------
        try:
            video_path = os.path.join(self.vis_root, vname + ".mkv")
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Missing or empty video file: {video_path}")

            video = self.vis_processor(video_path)
            if video is None or not torch.is_tensor(video):
                raise ValueError(f"vis_processor returned None or invalid tensor for {video_path}")

            valid_video = 1

        except Exception as e:
            print(f"[WARN] Failed to load video {vname}: {e}")
            # → bruit visuel aléatoire (8 frames RGB 224x224)
            video = torch.rand(3, 8, 224, 224)
            valid_video = 0

        # ------------------------
        #   AUDIO
        # ------------------------
        try:
            audio_file = os.path.join(
                self.audio_root, os.path.splitext(os.path.basename(vname))[0]
            ) + ".wav"

            waveform, sr = sf.read(audio_file, dtype="float32")
            if waveform is None or len(waveform) == 0:
                raise ValueError(f"Empty or None waveform in {audio_file}")

            # convertir en mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            waveform = torch.tensor(waveform).unsqueeze(0)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            valid_audio = 1

        except Exception as e:
            print(f"[WARN] Failed to load audio {vname}: {e}")
            # → bruit audio aléatoire
            waveform = torch.randn(1, 160125) * 1e-3
            valid_audio = 0

        # ------------------------
        #   PADDING & MASQUES
        # ------------------------
        audio, mask_BEATs, mask_LLM = pad_or_truncate_audio(waveform, valid_audio)

        # ------------------------
        #   VÉRIF FINALE
        # ------------------------
        if any(x is None for x in [video, audio, mask_BEATs, mask_LLM]):
            raise RuntimeError(f"[FATAL] Found None in sample {vname}")

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


# ================================================================
#   Dataset AudioCaps — Évaluation
# ================================================================
class VideoCaptionEvalDatasetAudioCaps(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.audio_root = "/home/abrimont/partage/AudioCaps/audiocaps_raw_audio/"
        self.vis_root = "/home/abrimont/partage/AudioCaps/audiocaps_raw_video/"

    def __getitem__(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        caption = ann.get("caption", "")
        image_id = ann.get("image_id", index)

        # ------------------------
        #   VIDÉO
        # ------------------------
        try:
            video_path = os.path.join(self.vis_root, vname + ".mkv")
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Missing or empty video file: {video_path}")

            video = self.vis_processor(video_path)
            if video is None or not torch.is_tensor(video):
                raise ValueError(f"vis_processor returned None or invalid tensor for {video_path}")

            valid_video = 1

        except Exception as e:
            print(f"[WARN] Failed to load eval video {vname}: {e}")
            # → bruit visuel aléatoire
            video = torch.rand(3, 8, 224, 224)
            valid_video = 0

        # ------------------------
        #   AUDIO
        # ------------------------
        try:
            audio_file = os.path.join(
                self.audio_root, os.path.splitext(os.path.basename(vname))[0]
            ) + ".wav"

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
            print(f"[WARN] Failed to load eval audio {vname}: {e}")
            waveform = torch.randn(1, 160125) * 1e-3
            valid_audio = 0

        # ------------------------
        #   PADDING & MASQUES
        # ------------------------
        audio, mask_BEATs, mask_LLM = pad_or_truncate_audio(waveform, valid_audio)

        if any(x is None for x in [video, audio, mask_BEATs, mask_LLM]):
            raise RuntimeError(f"[FATAL] Found None in eval sample {vname}")

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
