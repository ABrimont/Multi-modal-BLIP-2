# A Multi-modal BLIP-2 Approach for Video Captioning

[![Conference](https://img.shields.io/badge/ICPR-2026-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> Accepted at **ICPR 2026**.

With minimal fine-tuning strategies, we achieve state-of-the-art results on common Video Captioning (VC) datasets such as MSR-VTT and Latest-VATEX, alongside competitive performance on AudioCaps.

## 📖 Overview 

Recent advances in video captioning rely heavily on Vision-Language Models (VLMs). However, perfectly aligning audio, visual, and textual information to generate naturally perceptive, human-aligned captions remains challenging.

Our **open-source** model introduces a novel **Multi-modal Q-Former** adapter inspired by BLIP-2. It is explicitly designed to perform early-stage cross-modal feature extraction, seamlessly bridging the gap between audio, visual, and language modalities.

### 🧠 Key Contributions & Architecture

* **Early Audio-Visual Fusion:** Our design fosters fine-grained interactions between audio and visual cues at early layers. Visual queries can extract complementary information from audio, and vice versa.
* **The Multi-modal Q-Former:** A 12-layer transformer that retains the core BLIP-2 architecture but introduces two distinct sets of learnable queries (all with a dimensionality of 768):
  * **32 Visual Queries:** Attending to dense, frozen visual features (257 × 1024) extracted from ViT-L/14.
  * **16 Audio Queries:** Attending to frozen audio features (1500 × 768) extracted from BEATs.
* **Smart Feature Interaction:** Within the self-attention modules, attention is computed over the *concatenation* of audio and visual queries to force cross-modal interaction. During cross-attention, the unified multi-modal queries attend to both visual and audio encodings.
* **Massive Feature Compression:** The architecture efficiently compresses thousands of raw visual and audio features into a highly compact set of just **48 multi-modal tokens** that capture the most salient information.
* **Highly Efficient Training:** Demonstrates robust scalability by achieving state-of-the-art performance on vision-centric benchmarks (MSR-VTT, Latest-VATEX) and competitive results on audio-centric tasks (AudioCaps).

---

## ⚙️ Installation

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🗄️ Data Preparation

To download the necessary datasets, please refer to the official sources below:

* **MSR-VTT:** Available on [Kaggle](https://www.kaggle.com/datasets/vishnutheepb/msrvtt/data).
* **Latest-VATEX:** Since the original VATEX dataset was not fully available online during training, we use Latest-VATEX. The full dataset is available on [Hugging Face](https://huggingface.co/datasets/lmms-lab/VATEX).Latest-VATEX ids will be shared.
* **AudioCaps:** Available on the official [GitHub repository](https://audiocaps.github.io/).

Once downloaded, organize the videos, audio files, and annotations into the following directory structure:

```text
data/
└── dataset_name/
    ├── raw_videos/    # Put your raw .mp4/.avi files here
    ├── raw_audios/    # Put your extracted audio files here
    └── annotations/   # Put your annotation JSON/CSV files here
```

## 🚀 Getting Started

*(To be completed)*

## 🎓 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{...,
  title={A Multi-modal BLIP-2 Approach for Video Captioning},
  author={...},
  booktitle={ICPR},
  year={2026}
}
```
