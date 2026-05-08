import os
import yaml
from swift.llm import (
    get_model_tokenizer, load_dataset, get_template, EncodePreprocessor, get_model_arch,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
import torch
from datetime import datetime

logger = get_logger()
seed_everything(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config
with open(os.path.join(PROJECT_ROOT, 'config/sft_lora.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

model_id_or_path = cfg['model_path']
max_length = cfg['max_length']
lora_rank = cfg['lora_rank']
lora_alpha = cfg['lora_alpha']
freeze_llm = cfg['freeze_llm']
freeze_vit = cfg['freeze_vit']
freeze_aligner = cfg['freeze_aligner']
targs = cfg['training_args']
deepspeed_stage = cfg.get('deepspeed_stage', 2)
deepspeed_config = os.path.join(PROJECT_ROOT, f'DeepSpeed_config/deepspeed_zero{deepspeed_stage}.json')

system = (
    "Please answer this question based on the visual content. Provide your thinking process between the <think> and</think> tags, and then give your final answer between the <answer> and </answer> tags. At the end, you must output the final answer in the format: <answer><your_answer_here></answer>"
)

current_time = datetime.now().strftime("%m%d_%H")
output_dir = os.path.join(PROJECT_ROOT, cfg['output_dir'], current_time)

os.makedirs(output_dir, exist_ok=True)

dataset = [os.path.join(PROJECT_ROOT, cfg['dataset'])]
data_seed = 42
split_dataset_ratio = 0
num_proc = 4

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    deepspeed=deepspeed_config,
    learning_rate=targs['learning_rate'],
    per_device_train_batch_size=targs['per_device_train_batch_size'],
    per_device_eval_batch_size=1,
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
    dataloader_num_workers=16,
    data_seed=data_seed,
    remove_unused_columns=False,
    group_by_length=False,
    bf16=True,
    tf32=True,
    dataloader_pin_memory=True,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))

model, processor = get_model_tokenizer(model_id_or_path, model_type="qwen3_vl", model_kwargs={"attn_implementation": "flash_attention_2", 'device_map': None})

template = get_template(model.model_meta.template, processor, default_system=system, max_length=max_length, remove_unused_columns=False)
template.set_mode('train')
if template.use_model:
    template.model = model

target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit,
                            freeze_aligner=freeze_aligner)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)

train_dataset, _ = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
                                          seed=data_seed)

train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=data_seed)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    template=template,
)
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
