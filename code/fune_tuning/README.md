# Fine-tuning on Downstream Datasets

## Step 0: Prepare OmniVL-Guard Checkpoint

| Model | HuggingFace |
| :--- | :--- |
| **OmniVL-Guard-2B** | [SJJ0854/OmniVL-Guard-2B](https://huggingface.co/SJJ0854/OmniVL-Guard-2B) |
| **OmniVL-Guard-8B** | TBD |

## Step 1: Prepare Data

### SFT Data (`Datasets/Downstream-SFT.jsonl`)

Each line is a JSON with `messages` (question + answer), `images`/`videos`, and task metadata.

```json
{"messages": [{"role": "user", "content": "<image> Please determine if the image is Real or Fake."}, {"role": "assistant", "content": "<think>...</think>\n\n<answer>Answer: Fake.</answer>"}], "images": ["path/to/image.jpg"], "task": "2_cls", "type": "fake", "modality": "image"}
```

Key fields:
- `messages`: conversation with `user` (question) and `assistant` (answer with `<think>` and `<answer>` tags)
- `images` or `videos`: media file paths (relative to project root)
- `task`: task type (`2_cls`, `image_grounding`, `text_grounding`, `Grouding`)

### RL Data (`Datasets/Downstream-RL.jsonl`)

Each line is a JSON with `messages` (question only, no answer), `images`/`videos`, `label`, and task metadata.

```json
{"messages": [{"role": "user", "content": "<image> Please determine if the image is Real or Fake."}], "images": ["path/to/image.jpg"], "label": [1], "task": "2_cls", "type": "fake", "modality": "image"}
```

Key fields:
- `messages`: only the `user` message (no `assistant` answer — the model generates it during RL)
- `images` or `videos`: media file paths
- `label`: ground truth for reward computation
  - `2_cls`: `[0]` (Real) or `[1]` (Fake)
  - `image_grounding`: `[x1, y1, x2, y2]` bounding box
  - `text_grounding`: `[0, 3, 5]` tampered word indices
  - `Grouding`: `[start1, end1, start2, end2]` temporal segments in seconds
- `task`: task type

## Step 2: Fine-tune

Choose one of the following:

### Option A: SFT Only

Edit `config/finetune_sft.yaml` to set `model_path` (your OmniVL-Guard checkpoint) and `dataset`, then:

```bash
deepspeed --include localhost:0,1,2,3 code/fune_tuning/finetune_sft.py
```

### Option B: RL Only

Edit `config/finetune_rl.yaml` to set `model_path` (your OmniVL-Guard checkpoint) and `dataset`, then:

```bash
deepspeed --include localhost:0,1,2,3 code/fune_tuning/finetune_rl.py
```

### Option C: SFT then RL

1. Run SFT as Option A above.
2. Merge LoRA: edit `code/merge.sh` to set your fine-tuned checkpoint path, then `bash code/merge.sh`.
3. Edit `config/finetune_rl.yaml` to set `model_path` to the merged checkpoint, then run RL as Option B above.
