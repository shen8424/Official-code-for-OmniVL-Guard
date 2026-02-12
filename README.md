<div align="center">

# OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL

<a href="https://arxiv.org/abs/2602.10687"><img src="https://img.shields.io/badge/Paper-arXiv:2602.10687-b31b1b.svg" alt="arXiv"></a>
<a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<a href="#"><img src="https://img.shields.io/badge/Status-Under%20Review-yellow" alt="Status"></a>

</div>

---

## ⚠️ Status: Under Review & Coming Soon

> **Note:** This paper is currently under review.
>
> The full source code, model checkpoints, and the **FSFR (Full-Spectrum Forensic Reasoning)** dataset are currently being prepared for release. They will be made publicly available in this repository immediately following the review process. 
>
> Please **Star** ⭐ this repository to stay updated!

---

## 📖 Introduction

**OmniVL-Guard** is a unified framework designed to bridge this gap. It is the first framework capable of simultaneously handling forgery detection and grounding across dominant social media modalities (Image, Text, Video, and Image-Text) within a single paradigm.

### 🌟 Key Features
* **Unified Multi-Modal Defense:** Handles Text, Image, Video, and Image-Text forgeries simultaneously.
* **Balanced Reinforcement Learning:** Introduces **ARSPO** (Adaptive Reward Scaling Policy Optimization) to solve the "difficulty bias" in multi-task learning.
* **Reasoning-Driven:** Leverages **Self-Evolving CoT (Chain-of-Thought)** generation to synthesize high-quality reasoning paths, overcoming the cold-start challenge.
* **State-of-the-Art Performance:** Demonstrates superior performance on in-domain tests and robust zero-shot generalization on out-of-domain benchmarks.

---

## 🖼️ Framework Overview

<div align="center">
  <img src="assets/figure1_overview.png" alt="OmniVL-Guard Overview" width="100%">
  <br>
  <em>Figure 1: The unified vision-language forgery detection and grounding framework (OmniVL-Guard). The right side illustrates how ARSPO achieves balanced optimization compared to standard SFT.</em>
</div>

<br>

### 🧠 Core Components

1.  **Self-Evolving CoT Generation:**
    We propose a four-stage pipeline to generate high-quality forensic reasoning data:
    * *Source Data Collection* from diverse public datasets.
    * *Forensic Reasoning Seed Priming* using SOTA MLLMs.
    * *Seed Bootstrapping* via self-evolution.
    * *Collaborative Hard-CoT Synthesis* to handle long-tail hard samples.

2.  **ARSPO (Adaptive Reward Scaling Policy Optimization):**
    To address the imbalance where simple classification tasks dominate gradients, ARSPO dynamically modulates reward scales and task weights. It uses a **Task-Based Reward Mapping Function** (non-linear rewards for hard tasks) and **Dynamic Coefficient Adjustment** to ensure fine-grained localization tasks are learned effectively.

<div align="center">
  <img src="assets/figure2_pipeline.png" alt="CoT Generation Pipeline" width="100%">
  <br>
  <em>Figure 2: The Self-Evolving Forensic CoT Generation pipeline and statistics for the resulting FSFR dataset.</em>
</div>

---

## 📊 Dataset: FSFR

We present **FSFR (Full-Spectrum Forensic Reasoning)**, a comprehensive multimodal corpus designed for the full SFT-RL-Test pipeline.

* **Scale:** ~73k SFT samples (with CoT), ~110k RL samples, and ~700k Test samples.
* **Modalities:** Text, Image, Video, Image-Text.
* **Tasks:** * Binary Forgery Classification.
    * Tampering Localization (Spatial for Image, Semantic for Text, Temporal for Video).

*The download link and usage instructions for FSFR will be updated here.*

---

## 📈 Performance

OmniVL-Guard significantly outperforms existing SOTA MLLMs and domain-specific methods. Notably, it achieves massive gains in challenging localization tasks:

| Method | Binary Cls. | Image Loc. (IoU) | Text Loc. (F1) | Video Loc. (tIoU) |
| :--- | :---: | :---: | :---: | :---: |
| **OmniVL-Guard (Ours)** | **96.20%** | **54.26%** | **63.78%** | **59.22%** |
| Improvement vs Best | +6.97% | +5.73% | +22.92% | +37.79% |

*(See paper for full comparison tables).*

---

## 📝 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{shen2026omnivl,
  title={OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL},
  author={Shen, Jinjie and Wu, Jing and Wang, Yaxiong and Cheng, Lechao and Tang, Shengeng and Hui, Tianrui and Pu, Nan and Zhong, Zhun},
  journal={arXiv preprint arXiv:2602.10687},
  year={2026}
}
