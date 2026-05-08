import os
import re
from typing import TYPE_CHECKING, Dict, List, Union, Any
import string
import json
from math import exp
from collections import deque
import numpy as np
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from swift.llm import InferRequest

class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards



class DF_Format(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:        
        pattern = r'^<think>.*?</think>\n\n<answer>.*?</answer>(?![\s\S])'
        
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        
        return [0.5 if match else 0.0 for match in matches]


class AccuracyReward(ORM):
    # =========================================================================
    # [Class State]
    # =========================================================================

    _accumulated_stats: Dict[str, Dict] = {}

    _task_history_queue: Dict[str, deque] = {}

    _task_latest_total_deltas: Dict[str, float] = {}

    _task_coeffs: Dict[str, float] = {}
    _initial_baselines: Dict[str, float] = {}

    _warmup_collection_buffer: Dict[str, List[float]] = {}
    _baseline_locked = False

    _last_update_step = -1

    # =========================================================================
    # [Hyper Parameters]
    # =========================================================================
    
    UPDATE_INTERVAL = 400
    MIN_SAMPLES_TO_UPDATE = 4
    WINDOW_SIZE = 3

    WARMUP_STEPS = 12000
    BASELINE_COLLECT_START = 7000

    BOOST_RATE = 1.1             
    DECAY_RATE = 0.9             
    MAX_COEFF = 3.0             
    MIN_COEFF = 1.0              

    FAST_GROWTH_THRESHOLD = 0.02
    REGRESSION_THRESHOLD = -0.10
    TASK_HIGH_THRESHOLDS = {
        '2_cls': 0.07,
        'image_grounding': 0.50,
        'text_grounding': 0.60,
        'Grouding': 0.60
    }
    DEFAULT_HIGH_THRESH = 0.15

    KNOWN_TASKS = ['2_cls', 'image_grounding', 'text_grounding', 'Grouding']

    def __init__(self, log_file_path: str = "output/dynamic_reward_log.jsonl"):
        super().__init__()
        self.bbox_EXP_SCALE = 3.0
        self.textpos_EXP_SCALE = 3.0
        self.videogrounding_EXP_SCALE = 3.0
        
        self.bbox_denom = exp(self.bbox_EXP_SCALE) - 1
        self.text_denom = exp(self.textpos_EXP_SCALE) - 1
        self.video_denom = exp(self.videogrounding_EXP_SCALE) - 1

        self.log_file_path = log_file_path

        is_distributed = dist.is_initialized()
        if (not is_distributed) or (dist.get_rank() == 0):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        if not AccuracyReward._accumulated_stats:
            for task in self.KNOWN_TASKS:
                AccuracyReward._accumulated_stats[task] = {'sum': 0.0, 'count': 0}
                AccuracyReward._task_history_queue[task] = deque(maxlen=self.WINDOW_SIZE)
                AccuracyReward._task_coeffs[task] = 1.0
                AccuracyReward._warmup_collection_buffer[task] = []
                AccuracyReward._task_latest_total_deltas[task] = 0.0
                
                self._local_step_buffer = {t: [] for t in self.KNOWN_TASKS}

    # -------------------------------------------------------------------------
    # [Core Update Logic]
    # -------------------------------------------------------------------------
    def _flush_and_update(self, current_step):
        is_distributed = dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        device = torch.cuda.current_device()

        num_tasks = len(self.KNOWN_TASKS)
        local_tensor = torch.zeros(num_tasks * 2, device=device, dtype=torch.float32)

        for i, task in enumerate(self.KNOWN_TASKS):
            buffer = self._local_step_buffer[task]
            if buffer:
                local_tensor[i*2] = sum(buffer)
                local_tensor[i*2+1] = len(buffer)
            self._local_step_buffer[task] = [] 

        if is_distributed:
            dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
        
        global_stats = local_tensor.cpu().tolist()
        
        for i, task in enumerate(self.KNOWN_TASKS):
            g_sum = global_stats[i*2]
            g_count = int(global_stats[i*2+1])
            
            AccuracyReward._accumulated_stats[task]['sum'] += g_sum
            AccuracyReward._accumulated_stats[task]['count'] += g_count

            if self.BASELINE_COLLECT_START <= current_step < self.WARMUP_STEPS:
                if g_count > 0:
                    avg_val = g_sum / g_count
                    AccuracyReward._warmup_collection_buffer[task].append(avg_val)

        if current_step >= self.WARMUP_STEPS and not AccuracyReward._baseline_locked:
            
            for task in self.KNOWN_TASKS:
                collected_data = AccuracyReward._warmup_collection_buffer[task]
                if collected_data:
                    robust_avg = sum(collected_data) / len(collected_data)
                    AccuracyReward._initial_baselines[task] = max(robust_avg, 1e-6)
                else:
                    AccuracyReward._initial_baselines[task] = 1e-6
                
                AccuracyReward._warmup_collection_buffer[task] = [] 
            
            AccuracyReward._baseline_locked = True

        log_entry = {"step": current_step, "tasks": {}}
        candidates_for_laggard_boost = []

        for task in self.KNOWN_TASKS:
            stats = AccuracyReward._accumulated_stats[task]
            curr_count = stats['count']
            
            if curr_count < self.MIN_SAMPLES_TO_UPDATE:
                if rank == 0:
                    log_entry["tasks"][task] = {
                        "status": "accumulating_sparse", 
                        "samples": curr_count,
                        "coeff": round(AccuracyReward._task_coeffs[task], 2)
                    }
                continue
            
            curr_avg = stats['sum'] / curr_count

            history_q = AccuracyReward._task_history_queue[task]
            smooth_prev = sum(history_q) / len(history_q) if history_q else curr_avg
            
            AccuracyReward._task_history_queue[task].append(curr_avg)
            
            stats['sum'] = 0.0
            stats['count'] = 0

            initial_base = AccuracyReward._initial_baselines.get(task, max(smooth_prev, 1e-6))

            # -------------------------------------------------------------
            growth_abs_diff = curr_avg - smooth_prev
            delta_recent_relative = (curr_avg - smooth_prev) / max(smooth_prev, 1e-6)
            delta_total = (curr_avg - initial_base) / initial_base
            
            AccuracyReward._task_latest_total_deltas[task] = delta_total

            if rank == 0:
                log_entry["tasks"][task] = {
                    "metric": round(curr_avg, 4),
                    "growth_abs": round(growth_abs_diff, 5),
                    "delta_total": round(delta_total, 4),
                    "coeff": round(AccuracyReward._task_coeffs[task], 2)
                }

            if current_step < self.WARMUP_STEPS:
                continue

            # =================================================================
            # [Priority 1: Momentum Check]
            # =================================================================
            momentum_threshold = initial_base * self.FAST_GROWTH_THRESHOLD
            if growth_abs_diff > momentum_threshold:
                if rank == 0:
                    log_entry["tasks"][task]["action"] = "momentum_hold"
                continue 

            # =================================================================
            # [Priority 2: Rescue Check]
            # =================================================================
            if delta_recent_relative < self.REGRESSION_THRESHOLD:
                old_c = AccuracyReward._task_coeffs[task]
                new_c = min(old_c * self.BOOST_RATE, self.MAX_COEFF)
                AccuracyReward._task_coeffs[task] = new_c
                if rank == 0:
                    log_entry["tasks"][task]["action"] = "rescue_boost"
                continue 

            # =================================================================
            # [Priority 3: Balancing]
            # =================================================================

            task_specific_thresh = self.TASK_HIGH_THRESHOLDS.get(task, self.DEFAULT_HIGH_THRESH)
            
            is_declining = curr_avg < smooth_prev

            if (not is_declining) and (delta_total > task_specific_thresh):
                old_c = AccuracyReward._task_coeffs[task]
                new_c = max(old_c * self.DECAY_RATE, self.MIN_COEFF)
                if old_c != new_c:
                    AccuracyReward._task_coeffs[task] = new_c
                    if rank == 0:
                        log_entry["tasks"][task]["action"] = f"decay (thresh={task_specific_thresh})"
                continue

            candidates_for_laggard_boost.append(task)

        # ---------------------------------------------------------------------
        # [Global Laggard Boost]
        # ---------------------------------------------------------------------
        if current_step >= self.WARMUP_STEPS and candidates_for_laggard_boost:
            slowest_task = min(candidates_for_laggard_boost, key=lambda t: AccuracyReward._task_latest_total_deltas.get(t, 0.0))
            
            old_c = AccuracyReward._task_coeffs[slowest_task]
            if old_c < self.MAX_COEFF:
                new_c = min(old_c * self.BOOST_RATE, self.MAX_COEFF)
                AccuracyReward._task_coeffs[slowest_task] = new_c
                if rank == 0:
                    if slowest_task in log_entry["tasks"]:
                        log_entry["tasks"][slowest_task]["action"] = "laggard_boost"

        if rank == 0:
            self._write_log(log_entry)

    def _write_log(self, data: Dict):
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            print(f"[ERROR] Log write failed: {e}")

    # -------------------------------------------------------------------------
    # [Metric Helpers]
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_bbox_iou(box1: List[float], box2: List[float]) -> float:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top: return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        if union_area <= 0: return 0.0
        return intersection_area / union_area

    @staticmethod
    def _compute_f1(pred_indices: List[int], gt_indices: List[int]) -> float:
        pred_set = set(pred_indices)
        gt_set = set(gt_indices)
        if not gt_set and not pred_set: return 1.0
        if not gt_set or not pred_set: return 0.0
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) == 0: return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _merge_intervals(intervals: List[List[float]]) -> List[List[float]]:
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                merged.append(current)
        return merged

    def _compute_temporal_iou(self, pred_segs: List[List[float]], gt_segs: List[List[float]]) -> float:
        pred_merged = self._merge_intervals(pred_segs)
        gt_merged = self._merge_intervals(gt_segs)
        if not pred_merged and not gt_merged: return 1.0 
        if not pred_merged or not gt_merged: return 0.0  
        intersection_duration = 0.0
        for p in pred_merged:
            for g in gt_merged:
                start = max(p[0], g[0])
                end = min(p[1], g[1])
                if end > start: intersection_duration += (end - start)
        pred_duration = sum(p[1] - p[0] for p in pred_merged)
        gt_duration = sum(g[1] - g[0] for g in gt_merged)
        union_duration = pred_duration + gt_duration - intersection_duration
        if union_duration <= 1e-6: return 0.0
        return intersection_duration / union_duration

    # -------------------------------------------------------------------------
    # [Main Interface]
    # -------------------------------------------------------------------------
    def __call__(self, completions, **kwargs) -> List[float]:     
        tasks = kwargs.get('task', [])
        labels = kwargs.get('label', [])
        trainer_state = kwargs.get('trainer_state', None)
        current_step = trainer_state.global_step if trainer_state else 0
        
        rewards = []
        
        for completion, task, label_item in zip(completions, tasks, labels):
            raw_metric = 0.0
            norm_reward = 0.0

            # --- Task Metric Calculation ---
            if task == '2_cls':
                match = re.search(r'<answer>Answer:\s*(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).translate(str.maketrans('', '', string.punctuation)).strip().lower()
                    gt_val = label_item[0] if isinstance(label_item, list) else float(label_item)
                    if (abs(gt_val - 0.0) < 1e-5 and content == "real") or \
                       (abs(gt_val - 1.0) < 1e-5 and content == "fake"):
                        raw_metric = 1.0
                    norm_reward = raw_metric

            elif task == 'image_grounding':
                match = re.search(r'<answer>Tampered bbox:.*?\[(.*?)\].*?</answer>', completion, re.DOTALL)
                if match:
                    try:
                        pred_bbox = [float(x.strip()) for x in match.group(1).split(',')]
                        if len(pred_bbox) == 4:
                            iou = self._compute_bbox_iou(pred_bbox, label_item)
                            raw_metric = iou
                            norm_reward = (exp(iou * self.bbox_EXP_SCALE) - 1) / self.bbox_denom
                    except: pass

            elif task == 'text_grounding':
                match = re.search(r'<answer>Tampered words list:.*?\[(.*?)\].*?</answer>', completion, re.DOTALL)
                if match:
                    try:
                        content_str = match.group(1).strip()
                        pred_indices = [int(x.strip()) for x in content_str.split(',')] if content_str else []
                        f1 = self._compute_f1(pred_indices, label_item)
                        raw_metric = f1
                        norm_reward = (exp(f1 * self.textpos_EXP_SCALE) - 1) / self.text_denom
                    except: pass

            elif task == 'Grouding': 
                match = re.search(r'<answer>Tampered segment:\s*(\[.*?\])\s*</answer>', completion, re.DOTALL)
                if match:
                    try:
                        list_str = match.group(1)
                        seg_matches = re.findall(r"['\"]([\d\.]+)s-([\d\.]+)s['\"]", list_str)
                        pred_segs = []
                        for start_s, end_s in seg_matches:
                            s, e = float(start_s), float(end_s)
                            if e > s: pred_segs.append([s, e])
                        
                        raw_gt = label_item
                        gt_segs = []
                        if len(raw_gt) >= 2 and len(raw_gt) % 2 == 0:
                            for i in range(0, len(raw_gt), 2):
                                gt_segs.append([raw_gt[i], raw_gt[i+1]])
                        
                        tiou = self._compute_temporal_iou(pred_segs, gt_segs)
                        raw_metric = tiou
                        norm_reward = (exp(tiou * self.videogrounding_EXP_SCALE) - 1) / self.video_denom
                    except: pass

            if task in self.KNOWN_TASKS:
                self._local_step_buffer[task].append(raw_metric)

            coeff = AccuracyReward._task_coeffs.get(task, 1.0)
            rewards.append(float(norm_reward * coeff))
        
        if (current_step > 0 and
            current_step % self.UPDATE_INTERVAL == 0 and 
            current_step != AccuracyReward._last_update_step):
            
            self._flush_and_update(current_step)
            AccuracyReward._last_update_step = current_step

        return rewards




orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'df_format': DF_Format,
    'acc_reward': AccuracyReward,
}
