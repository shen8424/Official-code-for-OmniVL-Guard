# Training

## Step 0: Prepare Pretrained Model & Data

### Model

Download the base model from HuggingFace:

| Model | HuggingFace |
| :--- | :--- |
| **Qwen3-VL-2B-Instruct** | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| **Qwen3-VL-8B-Instruct** | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |

### Data

Download the FSFR dataset from [SJJ0854/FSFR](https://huggingface.co/datasets/SJJ0854/FSFR), then:

1. Extract `media_data.zip` and `media_data_RL.zip` to the project root directory.
2. Place the two `.jsonl` files under `./Datasets/`.

## Step 1: SFT

Edit `config/sft_lora.yaml` or `config/sft_full.yaml` to set `model_path` and `dataset`, then choose one:

```bash
# LoRA fine-tuning
bash code/sft.sh sft_lora

# Full-parameter fine-tuning
bash code/sft.sh sft_full
```

## Step 2: Merge LoRA

Only if you used `sft_lora`. Edit `code/merge.sh` to set your checkpoint path, then:

```bash
bash code/merge.sh
```

## Step 3: RL (ARSPO)

Edit `config/rl_lora.yaml` or `config/rl_full.yaml` to set `model_path` to your SFT checkpoint, then choose one:

```bash
# LoRA RL
bash code/rl.sh rl_lora

# Full-parameter RL
bash code/rl.sh rl_full
```

## Notes

- GPU count is specified in `code/sft.sh` and `code/rl.sh` via `--include localhost:0,1,2,3`. Adjust as needed.
