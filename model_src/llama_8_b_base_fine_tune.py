#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QLoRA SFT training for product price prediction.
- Stable for RTX A6000 (bf16)
- Supports WandB logging
- Dataset cleaning to prevent NaN losses
- Callable `train_sft()` function for Jupyter
"""

# =======================================================
# Imports
# =======================================================
import os
from datetime import datetime
import torch
import wandb
from dotenv import load_dotenv

from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,   # ✅ correct location
)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =======================================================
# Training function
# =======================================================
def train_sft(
    hf_user="Arupreza",
    dataset_name="ed-donner/pricer-data",
    base_model="meta-llama/Meta-Llama-3.1-8B",
    project_name="price_llama_lora",
    log_to_wandb=True,
    epochs=1,
    batch_size=4,
    save_steps=2000,
    run_name=None,
    push_to_hub=False,
    train_size=None,
    eval_size=None,
):
    """
    Train a QLoRA-SFT model for product price prediction.

    Returns:
        trainer, model, tokenizer, dataset
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

    # Hugging Face login
    hf_token = get_env_var("HF_TOKEN")
    login(hf_token, add_to_git_credential=True)

    # WandB login
    if log_to_wandb:
        wandb_api_key = get_env_var("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key)

    # ---------------------------
    # Constants
    # ---------------------------
    MAX_SEQUENCE_LENGTH = 182
    RUN_NAME = run_name or f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
    PROJECT_RUN_NAME = f"{project_name}-{RUN_NAME}"
    HUB_MODEL_NAME = f"{hf_user}/{PROJECT_RUN_NAME}"

    # LoRA params
    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.1
    QUANT_4_BIT = True

    # ---------------------------
    # Dataset
    # ---------------------------
    dataset = load_dataset(dataset_name)

    # clean dataset (avoid NaNs)
    def clean_example(example):
        text = example.get("text", "")
        if text is None or text.strip() == "":
            example["text"] = "N/A"
        return example

    dataset = dataset.map(clean_example)
    train = dataset["train"]
    test = dataset["test"]

    # Subset for debugging
    if train_size:
        train = train.select(range(min(train_size, len(train))))
    if eval_size:
        test = test.select(range(min(eval_size, len(test))))

    # ---------------------------
    # Quantization config
    # ---------------------------
    if QUANT_4_BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )

    # ---------------------------
    # Load model & tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")

    # ---------------------------
    # Collator (safer default)
    # ---------------------------
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---------------------------
    # LoRA config
    # ---------------------------
    lora_cfg = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    # ---------------------------
    # Training config
    # ---------------------------
    train_cfg = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        eval_strategy="steps",      # ✅ correct key
        eval_steps=200,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        save_total_limit=10,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb" if log_to_wandb else None,
        run_name=RUN_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dataset_text_field="text",
        save_strategy="steps",
        hub_strategy="every_save",
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

    # ---------------------------
    # Train
    # ---------------------------
    trainer.train()

    # ---------------------------
    # Push
    # ---------------------------
    if push_to_hub:
        trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)
        print(f"✅ Saved to Hugging Face Hub: {PROJECT_RUN_NAME}")

    if log_to_wandb:
        wandb.finish()

    return trainer, model, tokenizer, dataset