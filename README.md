# A Multi-modal BLIP-2 Approach for Video Captioning

[![Conference](https://img.shields.io/badge/ICPR-2026-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> Accepted at **ICPR 2026**.

This paper presents a novel multi-modal approach to video captioning that achieves state-of-the-art results on major benchmarks including MSR-VTT and Latest-VATEX, while maintaining competitive performance on AudioCaps, using minimal fine-tuning strategies.

## 📖 Overview 

Recent advances in video captioning have been driven by the success of Vision-Language Models (VLMs). However, effectively aligning audio, visual, and textual modalities to generate human-aligned, natural video captions remains a significant challenge.

Our work introduces an **open-source Multi-modal Q-Former** adapter inspired by BLIP-2, specifically designed for early-stage cross-modal feature fusion. This architecture seamlessly bridges visual and audio information through a novel query-based mechanism that preserves fine-grained modal interactions.

### 🧠 Key Contributions & Architecture

![Overview of the multi-modal BLIP-2 framework](projects/Figure_2.jpg)

* **Early Audio-Visual Fusion:** Our design promotes fine-grained interactions between audio and visual cues at early processing stages. Visual queries extract complementary information from audio features, and vice versa, enabling deeper cross-modal understanding.

* **The Multi-modal Q-Former:** A 12-layer transformer that builds upon the core BLIP-2 architecture while introducing two distinct learnable query sets (each with 768-dimensional embeddings):
  * **32 Visual Queries:** Attend to dense, frozen visual features (257 × 1024) extracted from ViT-L/14.
  * **16 Audio Queries:** Attend to frozen audio features (1500 × 768) extracted from BEATs.

* **Smart Feature Interaction:** Cross-modal interaction is enforced within self-attention modules through attention computed over concatenated audio and visual queries. During cross-attention, each modality queries the other modality's features, enabling bidirectional information flow.

* **Massive Feature Compression:** The architecture efficiently condenses thousands of raw visual and audio features into a compact 48-token multi-modal representation that captures the most salient information from both modalities.

* **Highly Efficient Training:** Demonstrates strong scalability across diverse benchmarks, achieving state-of-the-art performance on vision-centric datasets (MSR-VTT, Latest-VATEX) while maintaining competitive results on audio-centric benchmarks (AudioCaps).

---

## ⚙️ Installation

```bash
git clone https://github.com/ABrimont/Multi-modal-BLIP-2.git
cd Multi-modal-BLIP-2
sh setup.sh
```

Additionally, download the BEATs pre-trained weights (BEATs_iter3_plus_AS2M) from the [official repository](https://github.com/microsoft/unilm/tree/master/beats) and place them in the `weights/` directory.

---

## 📊 Performance Summary

**Multi-modal BLIP-2 Results:**

| Dataset  | CIDEr | METEOR | ROUGE-L | BLEU-4  | SPICE |
|---------|--------|--------|---------|---------|---------|
| **MSR-VTT** | 80.1 | 32.9 | 69.5 | 55.0 | 9.3 |
| **Latest-VATEX** | 86.8 | 28.8 | 56.7 | 44.2 | 15.0 |
| **AudioCaps** | 82.5 | 26.0 | 51.1 | 27.7 | 18.6 |

---

## 🗄️ Data Preparation & Evaluation

### Downloading Datasets

To obtain the necessary datasets, please visit the official sources below:

* **MSR-VTT:** Available on [Kaggle](https://www.kaggle.com/datasets/vishnutheepb/msrvtt/data)
* **Latest-VATEX:** The VATEX dataset is hosted on [Hugging Face](https://huggingface.co/datasets) - Latest-VATEX ids can be found here for train/val/test: (https://huggingface.co/datasets/AntBri/vatex-ids)
* **AudioCaps:** Available on the [official repository](https://audiocaps.github.io/)

### Organizing Raw Data

After downloading the datasets, organize them into the following directory structure:

```text
data/
└── dataset_name/
    ├── raw_videos/               # Raw video files (.mp4, .mkv, etc.)
    ├── raw_audios/               # Extracted audio files (.wav)
    ├── train.json                # Training annotations
    ├── val.json                  # Validation annotations
    ├── test.json                 # Test annotations
    └── *_gt.json                 # Ground truth in COCO format
```

where `*_gt.json` represents `msrvtt_gt.json`, `vatex_gt.json`, or `audiocaps_gt.json` depending on the dataset.

Ground truth files should follow the COCO caption format:

```json
{
  "annotations": [
    {
      "image_id": 1,
      "id": 1,
      "caption": "Vehicles hum and vibrate as they rev their engines"
    },
    {
      "image_id": 2,
      "id": 2,
      "caption": "A car engine accelerating and revving while tires skid"
    }
  ]
}
```

---

## 💾 Pre-trained Model Weights

To automatically download pre-trained weights for all three Multi-modal BLIP-2 variants (trained on MSR-VTT, Latest-VATEX, and AudioCaps), execute:

```bash
python dw.py
```

This script will download and organize all model weights into the `weights/` directory.

---

## 🚀 Getting Started

### Training

To initiate training, run:

```bash
bash train.sh
```

Training parameters and configuration files can be customized in:

```
lavis/projects/blip2/
```

### Evaluation

To evaluate the model, execute:

```bash
bash evaluate.sh
```

Evaluation settings can be adjusted in the same configuration directory.

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{brimont2026multimodal,
  title={A Multi-modal BLIP-2 Approach for Video Captioning},
  author={Brimont, ...},
  booktitle={Proceedings of the 28th International Conference on Pattern Recognition (ICPR)},
  year={2026}
}
```

---

## 🙏 Acknowledgments

This work builds upon contributions from several important projects:

* **BLIP-2** – [Salesforce Research](https://github.com/salesforce/BLIP) (Li et al., 2023)
* **BEATs** – [Microsoft UnilM](https://github.com/microsoft/unilm/tree/master/beats) (Chen et al., 2023)
* **Pretrained Image-Text Models are Secretly Video Captioners** – [Repository](https://github.com/chunhuizng/mllm-video-captioner/tree/main) (Zhang et al., 2025)

We gratefully acknowledge the authors of these repositories and papers for their foundational contributions to the field.
