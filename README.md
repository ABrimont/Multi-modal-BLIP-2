# A Multi-modal BLIP-2 Approach for Video Captioning

[![Conference](https://img.shields.io/badge/ICPR-2026-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

Accepted at ICPR 2026

We propose a multi-modal BLIP-2-based architecture for video captioning that achieves state-of-the-art performance on standard benchmarks such as MSR-VTT and VATEX, and competitive results on AudioCaps, while requiring only minimal fine-tuning.

--------------------------------------------------

OVERVIEW

Recent advances in video captioning rely heavily on Vision-Language Models (VLMs). However, effectively aligning audio, visual, and textual modalities to generate natural, human-like captions remains a challenging problem.

Our open-source model introduces a novel Multi-modal Q-Former inspired by BLIP-2. It is specifically designed to perform early-stage cross-modal feature extraction, enabling efficient interaction between modalities without requiring expensive video-text pre-training.

Following the PIT-VC paradigm (Pretrained Image-Text Models are Secretly Video Captioners), our approach leverages knowledge transferred directly from BLIP-2’s image-text pre-training.

--------------------------------------------------

KEY CONTRIBUTIONS & ARCHITECTURE

- Early Audio-Visual Fusion  
  The model enables fine-grained interactions between audio and visual features at early layers. Visual queries can attend to audio representations and vice versa, improving cross-modal reasoning.

- Multi-modal Q-Former  
  A 12-layer transformer extending the BLIP-2 Q-Former with two sets of learnable queries (dimension 768):
  - 32 Visual Queries attending to frozen visual features (257 x 1024) extracted from ViT-L/14  
  - 16 Audio Queries attending to frozen audio features (1500 x 768) extracted from BEATs  

- Cross-modal Interaction Mechanism  
  Self-attention operates on the concatenation of audio and visual queries to enforce interaction.  
  Cross-attention and feed-forward layers process modalities independently to preserve modality-specific representations.

- Efficient Feature Compression  
  Thousands of raw features are compressed into only 48 multi-modal tokens, capturing the most salient information for caption generation.

- Efficient Training Strategy  
  Achieves strong performance on both vision-centric datasets (MSR-VTT, VATEX) and audio-centric datasets (AudioCaps) with minimal training cost.

--------------------------------------------------

INSTALLATION

python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

--------------------------------------------------

DATA

To download the datasets, refer to their official sources:

MSR-VTT:
https://www.kaggle.com/datasets/vishnutheepb/msrvtt/data

VATEX (latest version):
https://huggingface.co/datasets/lmms-lab/VATEX

AudioCaps:
https://audiocaps.github.io/

Note: The latest VATEX version is used because the original dataset was not fully available during training.

DATA STRUCTURE

data/
  dataset_name/
    raw_videos/
    raw_audios/
    annotations/

--------------------------------------------------

GETTING STARTED

(To be completed)

--------------------------------------------------

CITATION

@inproceedings{your2026blip2,
  title={A Multi-modal BLIP-2 Approach for Video Captioning},
  author={Your Name et al.},
  booktitle={ICPR},
  year={2026}
}

--------------------------------------------------

LICENSE

This project is released under the MIT License.
