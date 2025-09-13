#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module version of RoBERTa fine-tuning with PEFT (LoRA).
Importable in Jupyter: from model_src.roberta_base_fine_tune import train_model
"""

import os
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
import tqdm as notebook_tqdm

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from model_src.preprocessing import load_and_preprocess_dataset, tok



# ---------------------------
# Metrics
# ---------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

class MetricWrapper:
    def __init__(self, use_log1p=True):
        self.use_log1p = use_log1p

    def __call__(self, eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        if self.use_log1p:
            preds = np.expm1(preds)
            labels = np.expm1(labels)
        return {"rmse": rmse(labels, preds), "mae": mae(labels, preds), "r2": r2_score(labels, preds)}


# ---------------------------
# Training Function
# ---------------------------
def train_model(
    source: str = "ed-donner/pricer-data",  # CSV path or HF dataset name
    text_column: str = "description",
    label_column: str = "price",
    out_dir: str = "./price_roberta_lora",
    max_length: int = 256,
    use_log1p: bool = True,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    grad_accum: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    eval_steps: int = 500,
    save_steps: int = 500,
    patience: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list = ["query", "value"],
    fp16: bool = True,
):
    """
    Train RoBERTa with PEFT (LoRA) for regression.

    Returns:
        trainer, model, tokenizer, dataset
    """

    # ---------------------------
    # Hugging Face login
    # ---------------------------
    load_dotenv()
    hf_token = os.getenv("HF_API_KEY")
    if not hf_token:
        raise ValueError("❌ HF_API_KEY not found in .env file!")
    login(token=hf_token)
    


    set_seed(seed)
    dataset = load_and_preprocess_dataset(
        source=source,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        seed=seed
    )


    # ---------------------------
    # Model + LoRA
    # ---------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base",
        num_labels=1,
        problem_type="regression",
    )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)

    collator = DataCollatorWithPadding(tokenizer=tok)

    # ---------------------------
    # TrainingArguments
    # ---------------------------
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        fp16=fp16,
        logging_steps=50,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=MetricWrapper(use_log1p=use_log1p),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()

    print("Validation metrics:", trainer.evaluate(dataset["validation"]))
    print("Test metrics:", trainer.evaluate(dataset["test"]))

    trainer.save_model()
    print(f"✅ Model saved to {out_dir}")

    return trainer, model, tok, dataset


# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    train_model()