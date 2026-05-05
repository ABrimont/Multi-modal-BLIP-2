# A Multi-modal BLIP-2 Approach for Video Captioning

[![Conference](https://img.shields.io/badge/ICPR-2026-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> Accepted in **ICPR 2026**.

With minimal fine-tuning strategies, we achieve state-of-the-art results on common Video Captioning (VC) datasets such as MSR-VTT and VATEX, and competitive results on AudioCaps.

## 📖 Overview 

Recent advances in video captioning rely heavily on Vision-Language Models (VLMs). However, perfectly aligning audio, visual, and textual information to generate naturally perceptive, human-aligned captions remains a challenge. 

Our **open-source** model introduces a novel **Multi-modal Q-Former** adapter inspired by BLIP-2. It is explicitly designed to perform early-stage cross-modal feature extraction, seamlessly bridging the gap between distinct modalities without the need for expensive video-text pre-training. Following the PIT-VC paradigm (Pretrained Image-Text Models are Secretly Video Captioners), our architecture transfers knowledge directly from BLIP-2’s image-text pre-training.

### 🧠 Key Contributions & Architecture

* **Early Audio-Visual Fusion:** Our design fosters fine-grained interactions between audio and visual cues at early layers. Visual queries can extract complementary information from audio, and vice versa, allowing each modality to inform and refine the other for deeper cross-modal reasoning.
* **The Multi-modal Q-Former:** A 12-layer transformer that retains the core BLIP-2 architecture but introduces two distinct sets of learnable queries (all with a dimensionality of `768`):
  * **`32` Visual Queries:** Attending to dense, frozen visual features (`257 × 1024`) extracted from ViT-L/14.
  * **`16` Audio Queries:** Attending to frozen audio features (`1500 × 768`) extracted from BEATs.
* **Smart Feature Interaction:** Within the self-attention modules, attention is computed over the *concatenation* of audio and visual queries to force cross-modal interaction. During cross-attention and feed-forward stages, the queries are processed *independently* to preserve their distinct modality-specific natures.
* **Massive Feature Compression:** The architecture efficiently compresses thousands of raw visual and audio features into a highly compact set of just **48 multi-modal tokens** that capture the most salient information for caption generation.
* **Highly Efficient Training:** Demonstrates robust scalability by achieving state-of-the-art performance on vision-centric benchmarks (MSR-VTT, Latest-VATEX) and competitive results on audio-centric ones (AudioCaps) under a minimal training regime.

---

## ⚙️ Installation
*(À compléter)*

## 🚀 Getting Started
*(À compléter)*

## 🎓 Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{...,
  title={A Multi-modal BLIP-2 Approach for Video Captioning},
  author={...},
  booktitle={ICPR},
  year={2026}
}
