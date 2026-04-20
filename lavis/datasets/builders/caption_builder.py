"""
 Copyright (c) 2022, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)

from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)

from lavis.datasets.datasets.video_caption_datasets_audio import (
    VideoCaptionDatasetAudio,
    VideoCaptionEvalDatasetAudio,
)


from lavis.datasets.datasets.video_caption_datasets_audio_vatex import (
    VideoCaptionDatasetAudioVATEX,
    VideoCaptionEvalDatasetAudioVATEX,
)

from lavis.datasets.datasets.video_caption_AudioCaps import (
    VideoCaptionDatasetAudioCaps,
    VideoCaptionEvalDatasetAudioCaps,
)

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("coco_train")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_train.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }

@registry.register_builder("msrvtt_caption_audio")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDatasetAudio
    eval_dataset_cls = VideoCaptionEvalDatasetAudio

    DATASET_CONFIG_DICT = {
        "default": "/home/abrimont/partage/mllm-video-captioner/lavis/configs/datasets/msrvtt_audio/defaults_cap.yaml",
    }



@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("msvd_train")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_train.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }

@registry.register_builder("vatex_caption_audio")
class VATEXAudioCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDatasetAudioVATEX
    eval_dataset_cls = VideoCaptionEvalDatasetAudioVATEX

    DATASET_CONFIG_DICT = {
        "default": "/home/abrimont/partage/mllm-video-captioner/lavis/configs/datasets/vatex_audio/defaults_cap.yaml",
    }


@registry.register_builder("AudioCaps_caption")
class AudioCapsCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDatasetAudioCaps
    eval_dataset_cls = VideoCaptionEvalDatasetAudioCaps

    DATASET_CONFIG_DICT = {
        "default": "/home/abrimont/partage/mllm-video-captioner/lavis/configs/datasets/AudioCaps/defaults_cap.yaml",
    }