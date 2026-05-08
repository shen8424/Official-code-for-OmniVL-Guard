## 🏆 Status: Accepted by ICML 2026

> **OmniVL-Guard** has been accepted by **ICML 2026**.
>
> The source code, model checkpoints, training pipeline, and the **FSFR (Full-Spectrum Forensic Reasoning)** dataset are currently being prepared for public release.
>
> We are committed to making this project fully reproducible and easy to extend for future research in multimodal forgery detection and grounding.

---

## 🖼️ Framework Overview

<div align="center">
  <img src="../figures/tease.png" alt="OmniVL-Guard Overview" width="100%">
  <br>
  <em>Figure 1: The unified vision-language forgery detection and grounding framework, OmniVL-Guard. The right side illustrates how ARSPO achieves balanced optimization compared to standard SFT.</em>
</div>

<br>

---

## 🧠 Core Components

### 1. Self-Evolving CoT Generation

We propose a four-stage pipeline to generate high-quality forensic reasoning data:

1. **Source Data Collection** from diverse public datasets.
2. **Forensic Reasoning Seed Priming** using state-of-the-art MLLMs.
3. **Seed Bootstrapping** through self-evolution.
4. **Collaborative Hard-CoT Synthesis** for long-tail and difficult samples.

This pipeline enables OmniVL-Guard to learn not only whether content is forged, but also why and where the forgery occurs.

### 2. ARSPO: Adaptive Reward Scaling Policy Optimization

To address the imbalance where simple classification tasks can dominate gradients, we introduce **ARSPO**.

ARSPO dynamically modulates reward scales and task weights through:

- **Task-Based Reward Mapping Function:** Applies adaptive and non-linear rewards for harder grounding tasks.
- **Dynamic Coefficient Adjustment:** Balances optimization across classification and localization objectives.
- **Difficulty-Aware Learning:** Ensures that fine-grained localization tasks are effectively learned instead of being overwhelmed by easier binary classification signals.

<div align="center">
  <img src="../figures/dataset.png" alt="CoT Generation Pipeline" width="100%">
  <br>
  <em>Figure 2: The Self-Evolving Forensic CoT Generation pipeline and statistics for the resulting FSFR dataset.</em>
</div>

---

## 📊 Dataset: FSFR

We present **FSFR (Full-Spectrum Forensic Reasoning)**, a comprehensive multimodal corpus designed for the complete SFT-RL pipeline.

### Dataset Scale

- **~73K SFT samples** with Chain-of-Thought forensic reasoning.
- **~110K RL samples** for balanced reinforcement learning.

### Modalities

- **Text**
- **Image**
- **Video**

### Tasks

- **Binary Forgery Classification**
- **Tampering Localization**
  - Spatial localization for images
  - Semantic localization for text
  - Temporal localization for videos

> The download link and usage instructions for FSFR will be released soon.

---

## 📈 Performance

OmniVL-Guard significantly outperforms existing state-of-the-art MLLMs and domain-specific forensic methods.

Notably, it achieves substantial gains in challenging localization tasks.

| Method | Binary Cls. | Image Loc. (IoU) | Text Loc. (F1) | Video Loc. (tIoU) |
| :--- | :---: | :---: | :---: | :---: |
| **OmniVL-Guard (Ours)** | **96.20%** | **54.26%** | **63.78%** | **59.22%** |
| Improvement vs Best | +6.97% | +5.73% | +22.92% | +37.79% |

Please refer to our paper for full comparison tables and detailed experimental analysis.
