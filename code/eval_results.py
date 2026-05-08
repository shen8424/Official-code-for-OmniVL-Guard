import json
import re
import string
import sys
import os
from math import exp
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ Metric helpers (from orm.py) ============

def compute_bbox_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def compute_f1(pred_indices, gt_indices):
    pred_set = set(pred_indices)
    gt_set = set(gt_indices)
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged


def compute_temporal_iou(pred_segs, gt_segs):
    pred_merged = merge_intervals(pred_segs)
    gt_merged = merge_intervals(gt_segs)
    if not pred_merged and not gt_merged:
        return 1.0
    if not pred_merged or not gt_merged:
        return 0.0
    intersection_duration = 0.0
    for p in pred_merged:
        for g in gt_merged:
            start = max(p[0], g[0])
            end = min(p[1], g[1])
            if end > start:
                intersection_duration += (end - start)
    pred_duration = sum(p[1] - p[0] for p in pred_merged)
    gt_duration = sum(g[1] - g[0] for g in gt_merged)
    union_duration = pred_duration + gt_duration - intersection_duration
    if union_duration <= 1e-6:
        return 0.0
    return intersection_duration / union_duration


# ============ Per-sample evaluation (from AccuracyReward.__call__) ============

def evaluate_sample(completion, task, label_item):
    """Returns (raw_metric, format_ok)"""
    raw_metric = 0.0
    format_ok = False

    if task == '2_cls':
        match = re.search(r'<answer>Answer:\s*(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
        if match:
            format_ok = True
            content = match.group(1).translate(str.maketrans('', '', string.punctuation)).strip().lower()
            gt_val = label_item[0] if isinstance(label_item, list) else float(label_item)
            if (abs(gt_val - 0.0) < 1e-5 and content == "real") or \
               (abs(gt_val - 1.0) < 1e-5 and content == "fake"):
                raw_metric = 1.0

    elif task == 'image_grounding':
        match = re.search(r'<answer>Tampered bbox:.*?\[(.*?)\].*?</answer>', completion, re.DOTALL)
        if match:
            format_ok = True
            try:
                pred_bbox = [float(x.strip()) for x in match.group(1).split(',')]
                if len(pred_bbox) == 4:
                    raw_metric = compute_bbox_iou(pred_bbox, label_item)
            except:
                pass

    elif task == 'text_grounding':
        match = re.search(r'<answer>Tampered words list:.*?\[(.*?)\].*?</answer>', completion, re.DOTALL)
        if match:
            format_ok = True
            try:
                content_str = match.group(1).strip()
                pred_indices = [int(x.strip()) for x in content_str.split(',')] if content_str else []
                raw_metric = compute_f1(pred_indices, label_item)
            except:
                pass

    elif task == 'Grouding':
        match = re.search(r'<answer>Tampered segment:\s*(\[.*?\])\s*</answer>', completion, re.DOTALL)
        if match:
            format_ok = True
            try:
                list_str = match.group(1)
                seg_matches = re.findall(r"['\"]([\d\.]+)s-([\d\.]+)s['\"]", list_str)
                pred_segs = []
                for start_s, end_s in seg_matches:
                    s, e = float(start_s), float(end_s)
                    if e > s:
                        pred_segs.append([s, e])
                raw_gt = label_item
                gt_segs = []
                if len(raw_gt) >= 2 and len(raw_gt) % 2 == 0:
                    for i in range(0, len(raw_gt), 2):
                        gt_segs.append([raw_gt[i], raw_gt[i + 1]])
                raw_metric = compute_temporal_iou(pred_segs, gt_segs)
            except:
                pass

    return raw_metric, format_ok


# ============ Main ============

def main():
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = os.path.join(PROJECT_ROOT, "output/results.jsonl")

    if not os.path.exists(results_path):
        print(f"Error: results file not found: {results_path}")
        print(f"Usage: python code/eval_results.py <results.jsonl>")
        return

    task_metrics = defaultdict(list)
    task_format_fail = defaultdict(int)
    task_total = defaultdict(int)

    cls_modality_metrics = defaultdict(list)
    cls_modality_format_fail = defaultdict(int)

    with open(results_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            task = obj.get('task', 'unknown')
            modality = obj.get('modality', 'unknown')
            label = obj.get('label', [])
            completion = obj.get('response', '')
            if not completion:
                for msg in obj.get('messages', []):
                    if msg.get('role') == 'assistant':
                        completion = msg.get('content', '')
                        break

            task_total[task] += 1
            raw_metric, format_ok = evaluate_sample(completion, task, label)
            task_metrics[task].append(raw_metric)
            if not format_ok:
                task_format_fail[task] += 1

            if task == '2_cls':
                cls_modality_metrics[modality].append(raw_metric)
                if not format_ok:
                    cls_modality_format_fail[modality] += 1

    # ============ Print results ============
    print("=" * 80)
    print(f"{'Task':<20} {'Count':>6} {'Metric':>10} {'FmtFail':>10} {'FmtFailRate':>12}")
    print("=" * 80)

    all_metrics = []
    for task in sorted(task_metrics.keys()):
        metrics = task_metrics[task]
        count = len(metrics)
        avg_metric = sum(metrics) / count if count > 0 else 0.0
        fmt_fail = task_format_fail[task]
        fmt_rate = fmt_fail / count if count > 0 else 0.0
        all_metrics.extend(metrics)
        print(f"{task:<20} {count:>6} {avg_metric:>10.4f} {fmt_fail:>10} {fmt_rate:>12.2%}")

    print("-" * 80)
    overall = sum(all_metrics) / len(all_metrics) if all_metrics else 0.0
    total_fmt_fail = sum(task_format_fail.values())
    print(f"{'OVERALL':<20} {len(all_metrics):>6} {overall:>10.4f} {total_fmt_fail:>10}")
    print("=" * 80)

    # ============ 2_cls breakdown by modality ============
    print("\n" + "=" * 80)
    print("[2_cls Breakdown by Modality]")
    print("=" * 80)
    print(f"{'Modality':<20} {'Count':>6} {'Accuracy':>10} {'FmtFail':>10} {'FmtFailRate':>12}")
    print("-" * 80)

    for modality in sorted(cls_modality_metrics.keys()):
        metrics = cls_modality_metrics[modality]
        count = len(metrics)
        correct = sum(1 for m in metrics if abs(m - 1.0) < 1e-6)
        accuracy = correct / count if count > 0 else 0.0
        fmt_fail = cls_modality_format_fail[modality]
        fmt_rate = fmt_fail / count if count > 0 else 0.0
        print(f"{modality:<20} {count:>6} {accuracy:>10.4f} {fmt_fail:>10} {fmt_rate:>12.2%}")

    print("=" * 80)

    # ============ Other tasks detail ============
    print("\n[Detail per task (non-2_cls)]")
    for task in sorted(task_metrics.keys()):
        if task == '2_cls':
            continue
        metrics = task_metrics[task]
        count = len(metrics)
        nonzero = sum(1 for m in metrics if m > 0)
        avg = sum(metrics) / count if count > 0 else 0.0
        fmt_fail = task_format_fail[task]
        print(f"  {task}:")
        print(f"    Total samples  : {count}")
        print(f"    Non-zero score : {nonzero} ({nonzero/count:.2%})")
        print(f"    Avg metric     : {avg:.4f}")
        print(f"    Format failures: {fmt_fail} ({fmt_fail/count:.2%})")
        print()


if __name__ == '__main__':
    main()
