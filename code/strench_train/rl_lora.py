import os
import math
import yaml
from swift.llm import (
    get_model_tokenizer, load_dataset, get_template, EncodePreprocessor, get_model_arch,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import GRPOTrainer, GRPOConfig
import torch
from datetime import datetime


def round_to_nearest_100(x):
    return int(round(x / 100) * 100)

seed_everything(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config
with open(os.path.join(PROJECT_ROOT, 'config/rl_lora.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

model_id_or_path = os.path.join(PROJECT_ROOT, cfg['model_path'])
max_length = cfg['max_length']
lora_rank = cfg['lora_rank']
lora_alpha = cfg['lora_alpha']
freeze_llm = cfg['freeze_llm']
freeze_vit = cfg['freeze_vit']
freeze_aligner = cfg['freeze_aligner']
targs = cfg['training_args']
grpo = cfg['grpo']
cosine = cfg['cosine']
rep = cfg['repetition']
arspo = cfg['arspo']
deepspeed_stage = cfg.get('deepspeed_stage', 2)
deepspeed_config = os.path.join(PROJECT_ROOT, f'DeepSpeed_config/deepspeed_zero{deepspeed_stage}_rl.json')

system = (
    "Please answer this question based on the visual content. "
    "Provide your thinking process between the <think> and</think> tags, "
    "and then give your final answer between the <answer> and </answer> tags. "
    "At the end, you must output the final answer in the format: <answer><your_answer_here></answer>"
)

current_time = datetime.now().strftime("%m%d_%H")
output_dir = os.path.join(PROJECT_ROOT, cfg['output_dir'], current_time)

os.makedirs(output_dir, exist_ok=True)

dataset = [os.path.join(PROJECT_ROOT, cfg['dataset'])]
data_seed = 42
split_dataset_ratio = 0
num_proc = 16

training_args = GRPOConfig(
    output_dir=output_dir,
    deepspeed=deepspeed_config,
    learning_rate=targs['learning_rate'],
    per_device_train_batch_size=targs['per_device_train_batch_size'],
    per_device_eval_batch_size=4,
    gradient_checkpointing=targs['gradient_checkpointing'],
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=targs['save_steps'],
    eval_strategy='steps',
    eval_steps=100000,
    gradient_accumulation_steps=targs['gradient_accumulation_steps'],
    num_train_epochs=targs['num_train_epochs'],
    metric_for_best_model='loss',
    save_total_limit=targs['save_total_limit'],
    logging_steps=5,
    dataloader_num_workers=4,
    data_seed=data_seed,
    remove_unused_columns=False,
    # GRPO Arguments
    max_completion_length=grpo['max_completion_length'],
    epsilon=grpo['epsilon'],
    cosine_min_len_value_wrong=cosine['min_len_value_wrong'],
    cosine_max_len_value_wrong=cosine['max_len_value_wrong'],
    cosine_min_len_value_correct=cosine['min_len_value_correct'],
    cosine_max_len_value_correct=cosine['max_len_value_correct'],
    repetition_n_grams=rep['n_grams'],
    repetition_max_penalty=rep['max_penalty'],
    reward_model=None,
    num_generations=grpo['num_generations'],
    scale_rewards=grpo.get('scale_rewards', 'batch'),
    loss_type=grpo.get('loss_type', 'sapo'),
    beta=grpo['beta'],
    log_completions=True,
    use_vllm=grpo['use_vllm'],
    vllm_gpu_memory_utilization=grpo['vllm_gpu_memory_utilization'],
    vllm_max_model_len=max_length,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
model, processor = get_model_tokenizer(model_id_or_path, model_type="qwen3_vl", model_kwargs={"attn_implementation": "flash_attention_2", 'device_map': None})
ref_model = None
template = get_template(model.model_meta.template, processor, default_system=system, max_length=max_length, remove_unused_columns=False)
template.set_mode('train')
template.truncation_strategy = 'left'

if template.use_model:
    template.model = model

target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit,
                            freeze_aligner=freeze_aligner)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)

# Auto-calculate ARSPO hyperparameters from total training steps
# In GRPO, per_device_train_batch_size counts completions (rollouts), not prompts.
# Unique prompts per step = (per_device_bs / num_generations) * n_gpus * grad_accum
n_gpus = torch.cuda.device_count() or 1
n_samples = len(open(os.path.join(PROJECT_ROOT, cfg['dataset'])).readlines())
prompts_per_step = (targs['per_device_train_batch_size'] / grpo['num_generations']) * n_gpus * targs['gradient_accumulation_steps']
steps_per_epoch = math.ceil(n_samples / prompts_per_step)
total_steps = steps_per_epoch * targs['num_train_epochs']

arspo_update_interval = max(100, round_to_nearest_100(total_steps * arspo['update_interval_ratio']))
arspo_warmup_steps = round_to_nearest_100(total_steps * arspo['warmup_steps_ratio'])
arspo_baseline_collect_start = round_to_nearest_100(total_steps * arspo['baseline_collect_start_ratio'])

from swift.plugin.orm import AccuracyReward
AccuracyReward.UPDATE_INTERVAL = arspo_update_interval
AccuracyReward.WARMUP_STEPS = arspo_warmup_steps
AccuracyReward.BASELINE_COLLECT_START = arpo_baseline_collect_start

logger = get_logger()
logger.info(f"Total training steps: {total_steps}")
logger.info(f"ARSPO: UPDATE_INTERVAL={arspo_update_interval}, WARMUP_STEPS={arspo_warmup_steps}, BASELINE_COLLECT_START={arspo_baseline_collect_start}")

train_dataset, _ = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc, seed=data_seed, remove_unused_columns=False)

trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        reward_model=None,
        reward_funcs=['repetition', 'df_format', 'acc_reward'],
        train_dataset=train_dataset,
        template=template
    )
trainer.train()
last_model_checkpoint = trainer.state.last_model_checkpoint
