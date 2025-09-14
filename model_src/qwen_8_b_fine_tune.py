#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import Optional

import torch
from dotenv import load_dotenv

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM



def train_sft(
    # Identity
    project_name: str = "qwen_finetune_for_price_prediction",
    base_model: str = "Qwen/Qwen1.5-8B",

    # Data
    dataset_name: str = "ed-donner/pricer-data",
    text_column: str = "text",
    label_column: str = "price",
    max_seq_len: int = 512,  # Qwen can handle 32k, but 512 is safer/faster here

    # LoRA / QLoRA
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules=("c_attn", "c_proj", "w1", "w2", "w3"),
    quant_4bit: bool = True,

    # Trainer hparams
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum_steps: int = 8,   # effective batch = 64
    learning_rate: float = 2e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    optim_name: str = "paged_adamw_32bit",
    save_steps: int = 2000,
    save_total_limit: int = 10,
    logging_steps: int = 50,
    eval_steps: int = 1000,

    # Precision
    fp16: bool = True,
    bf16: bool = False,

    # WandB
    log_to_wandb: bool = True,

    # Output / naming
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,

    # Repro / misc
    seed: int = 42,
    response_template: str = "Price is $",
    resume_from_checkpoint: Optional[str] = None,
):
    # ------------------------------
    # Env & WandB
    # ------------------------------
    load_dotenv()
    if log_to_wandb:
        import wandb
        wandb_key = os.getenv("WANDB_API_KEY")
        if not wandb_key:
            raise ValueError("❌ WANDB_API_KEY not found in environment variables.")
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb.login()
        os.environ["WANDB_PROJECT"] = project_name
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # ------------------------------
    # IDs and names
    # ------------------------------
    if run_name is None:
        run_name = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    project_run_name = f"{project_name}-{run_name}"
    if output_dir is None:
        output_dir = project_run_name

    # ------------------------------
    # Data preprocessing
    # ------------------------------
    set_seed(seed)
    ds = load_and_preprocess_dataset(
        source=dataset_name,
        text_column=text_column,
        label_column=label_column,
        max_length=max_seq_len,
        seed=seed,
    )

    # ------------------------------
    # Quantization config
    # ------------------------------
    if quant_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )

    # ------------------------------
    # Tokenizer & base model
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base.generation_config.pad_token_id = tokenizer.pad_token_id

    # ------------------------------
    # LoRA config
    # ------------------------------
    lora_cfg = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )

    # ------------------------------
    # Data collator
    # ------------------------------
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ------------------------------
    # Training config
    # ------------------------------
    train_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=eval_steps,
        gradient_accumulation_steps=grad_accum_steps,
        optim=optim_name,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=0.3,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=("wandb" if log_to_wandb else None),
        run_name=run_name,
        max_seq_length=max_seq_len,
        dataset_text_field=text_column,
        save_strategy="steps",
        push_to_hub=False,   # ❌ No hub push
        hub_model_id=None,
        hub_private_repo=None,
        hub_strategy=None,
    )

    # ------------------------------
    # Build trainer
    # ------------------------------
    trainer = SFTTrainer(
        model=base,
        train_dataset=ds["train"].select(range(min(100000, len(ds["train"])))),
        eval_dataset=ds["validation"].select(range(min(1000, len(ds["validation"])))),
        peft_config=lora_cfg,
        args=train_cfg,
        data_collator=collator,
    )

    # ------------------------------
    # Train
    # ------------------------------
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"✅ Training finished. Model saved locally at {output_dir}")

    return trainer, trainer.model, tokenizer, ds