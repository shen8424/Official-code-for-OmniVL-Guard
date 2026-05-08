# Inference

## Step 1: Run Inference

Edit `code/test.sh` to set:

- `CKPT`: your model checkpoint path
- `RESULT_PATH`: output result path
- `--val_dataset`: test dataset path

Then:

```bash
bash code/test.sh
```

## Step 2: Evaluate

```bash
python code/eval_results.py ${RESULT_PATH}
```
