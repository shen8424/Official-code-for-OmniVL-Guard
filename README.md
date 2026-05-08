<div align="center">

# OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL

<a href="https://arxiv.org/abs/2602.10687"><img src="https://img.shields.io/badge/Paper-arXiv:2602.10687-b31b1b.svg" alt="arXiv"></a>
<a href="#"><img src="https://img.shields.io/badge/Conference-ICML%202026-4b8bbe.svg" alt="ICML 2026"></a>
<a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<a href="#"><img src="https://img.shields.io/badge/Status-Accepted-brightgreen.svg" alt="Status"></a>
<a href="https://huggingface.co/datasets/SJJ0854/FSFR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-FSFR-ff9900.svg" alt="HF Dataset"></a>
<a href="https://huggingface.co/SJJ0854/OmniVL-Guard-2B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-OmniVL--Guard--2B-ff9900.svg" alt="HF Model"></a>

</div>

---

## 🎉 News

- [x] **[2026.04.30]** OmniVL-Guard has been accepted to **ICML 2026**!
- [ ] Release **OmniVL-Guard-8B**, our flagship unified vision-language forensic model.
- [x] **[2026.05.08]** Release **OmniVL-Guard-2B**, a lightweight and efficient version for broader deployment.
- [x] **[2026.05.08]** Release **FSFR (Full-Spectrum Forensic Reasoning)**, a large-scale multimodal forensic reasoning dataset.
- [x] **[2026.05.08]** Open-source the complete **OmniVL-Guard code base**, including the full training pipeline from scratch.
- [x] **[2026.05.08]** Provide fine-tuning recipes and checkpoints for adapting **OmniVL-Guard-8B / OmniVL-Guard-3B** to additional forensic datasets.

Please **Star** ⭐ this repository to stay updated!

---

## 📖 Introduction

**OmniVL-Guard** is a unified framework designed to bridge the gap between multimodal forgery detection and fine-grained grounding.

It is the first framework capable of simultaneously handling forgery detection and grounding across dominant social media modalities — **Image, Text, and Video** — within a single paradigm.

### 🌟 Key Features

- **Unified Multi-Modal Defense:** Handles text, image, and video forgeries simultaneously.
- **Balanced Reinforcement Learning:** Introduces **ARSPO** — Adaptive Reward Scaling Policy Optimization — to address the “difficulty bias” in multi-task learning.
- **Reasoning-Driven Forensics:** Leverages **Self-Evolving CoT (Chain-of-Thought)** generation to synthesize high-quality forensic reasoning paths and overcome the cold-start challenge.
- **Fine-Grained Grounding:** Supports spatial localization for images, semantic localization for text, and temporal localization for videos.
- **State-of-the-Art Performance:** Achieves strong in-domain performance and robust zero-shot generalization on out-of-domain benchmarks.

> For more details about the framework, dataset, and performance, please refer to [Introduction](./Introduction/README.md).

---

---

## 🛠️ Quick Start

### 1. Environment Setup

```bash
conda create -n OmniVL-Guard python==3.10
conda activate OmniVL-Guard
pip install ms-swift==3.10.3
pip install decord
pip install vllm==0.11.0
pip install deepspeed==0.17.6
pip install qwen_vl_utils==0.0.14
```

Download the pre-compiled flash-attention wheel from [Google Drive](https://drive.google.com/file/d/1b1Gxcwb3E7ft5vRyoJM60x5Di707_kvf/view?usp=sharing), then install it:

```bash
pip install /path/to/flash_attn-*.whl
```

### 2. Choose Your Path

| Use Case | Guide |
| :--- | :--- |
| 🔧 **Fine-tune on OmniVL-Guard 2B/8B** | [Fine-tuning Guide](./code/fune_tuning/README.md) |
| 🚀 **Run Inference** | [Inference Guide](./code/Inference/README.md) |
| 🏗️ **Train from Scratch** | [Full Training Guide](./code/strench_train/README.md) |

---

## 📝 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{shen2026omnivl,
  title={OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL},
  author={Shen, Jinjie and Wu, Jing and Wang, Yaxiong and Cheng, Lechao and Tang, Shengeng and Hui, Tianrui and Pu, Nan and Zhong, Zhun},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```
