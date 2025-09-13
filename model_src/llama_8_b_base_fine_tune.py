#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SFT (QLoRA) training for product price prediction following the
'Week 7 Day 3: TRAINING' Colab structure, adapted for RTX A6000 stability.

Usage (Notebook):
    from model_src.llama_8_b_base_fine_tune import train_sft

    trainer, model, tokenizer, ds = train_sft(
        hf_user="ed-donner",
        dataset_name="ed-donner/pricer-data",
        base_model="meta-llama/Meta-Llama-3.1-8B",
        project_name="pricer",
        log_to_wandb=True,
        epochs=1,
        batch_size=4,
        save_steps=2000,
        run_name=None,
        push_to_hub=True
    )
"""

import os
from datetime import datetime
from typing import Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from model_src.llama_8_b_preprocessing import load_and_preprocess_dataset


def train_sft(
    # Admin / identity
    hf_user: str = "ed-donner",
    project_name: str = "llama_finetune_for_price_prediction_from_product_description",
    base_model: str = "meta-llama/Meta-Llama-3.1-8B",

    # Data
    dataset_name: str = "ed-donner/pricer-data",
    text_column: str = "text",
    label_column: str = "price",
    max_seq_len: int = 256,

    # LoRA / QLoRA
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    quant_4bit: bool = True,

    # Trainer hparams
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.03,
    optim_name: str = "paged_adamw_32bit",
    save_steps: int = 2000,
    save_total_limit: int = 10,
    logging_steps: int = 50,
    eval_steps: int = 500,

    # Precision
    fp16: bool = True,
    bf16: bool = False,  # RTX A6000: bf16 not stable

    # WandB / Hub
    log_to_wandb: bool = True,
    push_to_hub: bool = True,
    hub_private_repo: bool = True,

    # Output / naming
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,

    # Repro / misc
    seed: int = 42,
    response_template: str = "Price is $",
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Returns:
        trainer, model, tokenizer, dataset_dict
    """

    # ------------------------------
    # Env & login
    # ------------------------------
    load_dotenv()
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_KEY")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not hf_token:
        raise ValueError("❌ Hugging Face token not found in environment variables.")
    login(hf_token, add_to_git_credential=True)

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
    hub_model_name = f"{hf_user}/{project_run_name}"
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
        weight_decay=0.001,
        fp16=bool(fp16),
        bf16=bool(bf16),
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=("wandb" if log_to_wandb else None),
        run_name=run_name,
        max_seq_length=max_seq_len,
        dataset_text_field=text_column,
        save_strategy="steps",
        hub_strategy="every_save" if push_to_hub else "end",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_name if push_to_hub else None,
        hub_private_repo=hub_private_repo if push_to_hub else None,
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

    # ------------------------------
    # Push to Hub
    # ------------------------------
    if push_to_hub:
        trainer.model.push_to_hub(project_run_name, private=hub_private_repo)
        print(f"✅ Saved to the hub: {project_run_name}")

    return trainer, trainer.model, tokenizer, ds