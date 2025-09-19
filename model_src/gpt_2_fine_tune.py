#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT-2 SFT training for product price prediction.
- Supports WandB logging
- Dataset cleaning to prevent NaN losses
- Callable `train_sft()` function for Jupyter
"""

import os
from datetime import datetime
import torch
import wandb
from dotenv import load_dotenv

from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def train_sft(
    hf_user="Arupreza",
    dataset_name="ed-donner/pricer-data",
    base_model="gpt2",   # ✅ replaced with GPT-2
    project_name="price_gpt2_sft",
    log_to_wandb=True,
    epochs=3,
    batch_size=8,
    save_steps=500,
    run_name=None,
    push_to_hub=False,
    train_size=None,
    eval_size=None,
    use_lora=False,   # ✅ optional LoRA
):
    """
    Train a GPT-2 model (optionally with LoRA) for product price prediction.
    """

    # ---------------------------
    # Load environment variables
    # ---------------------------
    load_dotenv()

    def get_env_var(name, default=None):
        val = os.getenv(name)
        if val is None and default is None:
            raise ValueError(f"Environment variable {name} not set")
        return val if val is not None else default

    hf_token = get_env_var("HF_TOKEN")
    login(hf_token, add_to_git_credential=True)

    if log_to_wandb:
        wandb_api_key = get_env_var("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key)

    # ---------------------------
    # Constants
    # ---------------------------
    MAX_SEQUENCE_LENGTH = 512  # GPT-2 supports up to 1024
    RUN_NAME = run_name or f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    PROJECT_RUN_NAME = f"{project_name}-{RUN_NAME}"
    HUB_MODEL_NAME = f"{hf_user}/{PROJECT_RUN_NAME}"

    # ---------------------------
    # Dataset
    # ---------------------------
    dataset = load_dataset(dataset_name)

    def clean_example(example):
        text = example.get("text", "")
        if text is None or text.strip() == "":
            example["text"] = "N/A\nPrice is $0.00"
        elif "Price is $" not in text:
            example["text"] = text.strip() + "\nPrice is $"
        return example

    dataset = dataset.map(clean_example)
    train = dataset["train"]
    test = dataset["test"]

    if train_size:
        train = train.select(range(min(train_size, len(train))))
    if eval_size:
        test = test.select(range(min(eval_size, len(test))))

    # ---------------------------
    # Load model & tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    # ---------------------------
    # Collator
    # ---------------------------
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ---------------------------
    # LoRA config (optional)
    # ---------------------------
    lora_cfg = None
    if use_lora:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        )

    # ---------------------------
    # Training config
    # ---------------------------
    train_cfg = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=200,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_steps=save_steps,
        save_total_limit=5,
        logging_steps=50,
        learning_rate=5e-5,   # ✅ tuned for GPT-2
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb" if log_to_wandb else None,
        run_name=RUN_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dataset_text_field="text",
        save_strategy="steps",
        push_to_hub=push_to_hub,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
    )

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=test,
        peft_config=lora_cfg,
        args=train_cfg,
        data_collator=collator,
    )

    trainer.train()

    if push_to_hub:
        trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)
        print(f"✅ Saved to Hugging Face Hub: {PROJECT_RUN_NAME}")

    if log_to_wandb:
        wandb.finish()

    return trainer, model, tokenizer, dataset