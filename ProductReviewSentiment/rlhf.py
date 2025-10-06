# =====================================================================================
# SECTION 0: IMPORTS AND CONFIGURATION
# =====================================================================================
import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
    BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, RewardTrainer, PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from tqdm import tqdm
import numpy as np

# Anti-OOM env (set at top for safety)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables for secure token management
load_dotenv()

class ScriptConfig:
    """Centralized configuration for the entire pipeline."""
    # --- Model & Tokenizer ---
    BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MAX_SEQ_LENGTH = 1024  # Shorter for VRAM efficiency (reviews fit)

    # --- Datasets ---
    SFT_DATASET_ID = "amazon_polarity"
    RM_PPO_DATASET_PATH = "/home/lisa/Arupreza/LLM-Support-Tools/SFT_and_RLHF/RLHF_data_for_sentiment_product_review.json"

    # --- Paths for Saved Adapters ---
    SFT_ADAPTER_PATH = "./sft_llama3_adapters"
    RM_ADAPTER_PATH = "./rm_llama3_adapters"
    PPO_MODEL_PATH = "./ppo_llama3_model"

config = ScriptConfig()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Set HF_TOKEN in your .env file.")

# =====================================================================================
# STAGE 1: SUPERVISED FINE-TUNING (SFT)
# =====================================================================================

def run_sft():
    """
    PURPOSE: Teach the base model the instruction-following format.
    OUTPUT: A LoRA adapter containing the learned knowledge.
    """
    print("--- 🚀 STAGE 1: SUPERVISED FINE-TUNING (SFT) ---")

    # --- 1. Load Model and Tokenizer ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
    except Exception as e:
        print(f"Model loading failed: {e}. Check token/GPU.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load and Prepare Dataset ---
    full_train = load_dataset(config.SFT_DATASET_ID, split="train")
    train_dataset = full_train.select(range(int(len(full_train) * 0.01)))  # ~1% for dev (~18k)
    print(f"Training samples: {len(train_dataset)}")

    full_eval = load_dataset(config.SFT_DATASET_ID, split="test")
    eval_dataset = full_eval.select(range(int(len(full_eval) * 0.01)))  # ~1% (~2k)
    print(f"Eval samples: {len(eval_dataset)}")

    def format_review(examples):
        texts = []
        for label, content in zip(examples['label'], examples['content']):
            sentiment = "Positive" if label == 1 else "Negative"
            messages = [
                {"role": "user", "content": f"Analyze the sentiment of this Amazon product review and classify it as either Positive or Negative.\n\nReview:\n\"{content}\""},
                {"role": "assistant", "content": f"Sentiment: {sentiment}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length"  # <-- KEY CHANGE: Uniform lengths; collator chills
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()  # Shallow copy fine now (uniform)
        tokenized["attention_mask"] = tokenized["attention_mask"].copy()  # Ensure mask too
        return tokenized

    formatted_train_dataset = train_dataset.map(
        format_review, 
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset",
        batched=True,
        batch_size=64
    )
    formatted_eval_dataset = eval_dataset.map(
        format_review, 
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval dataset",
        batched=True,
        batch_size=64
    )

    # --- 3. Configure LoRA and Trainer ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=config.SFT_ADAPTER_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Smaller for VRAM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch=16
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False  # Bypasses token drama
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # bf16 alignment
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        peft_config=peft_config,
        data_collator=data_collator
    )

    # --- 4. Train and Save Adapter ---
    trainer.train()
    trainer.save_model()
    print(f"--- ✅ SFT Stage Complete. Best model adapter saved to {config.SFT_ADAPTER_PATH} ---")

# =====================================================================================
# STAGE 2: REWARD MODELING (RM)
# =====================================================================================
def run_reward_modeling():
    """
    PURPOSE: Train a classifier to predict which of two responses is better.
    OUTPUT: A LoRA adapter for the reward model.
    """
    print("\n--- 🚀 STAGE 2: REWARD MODELING (RM) ---")

    # --- 1. Load a New Base Model for Sequence Classification ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.BASE_MODEL_ID,
            num_labels=1,  # Regression for scalar rewards
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
    except Exception as e:
        print(f"RM Model loading failed: {e}. Check token/GPU.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 2. Load and Prepare Preference Dataset (Keep as Text for TRL) ---
    full_dataset = load_dataset("json", data_files=config.RM_PPO_DATASET_PATH, split="train")
    dataset_splits = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = dataset_splits['train']
    eval_dataset = dataset_splits['test']

    # Format: Add 'prompt' column; TRL handles tokenization of chosen/rejected
    def add_prompt(example):
        example['prompt'] = f"Provide a sentiment analysis summary for the following review:\n\nReview:\n\"{example['review']}\""
        return example

    train_dataset = train_dataset.map(add_prompt)
    eval_dataset = eval_dataset.map(add_prompt)

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # TRL provides (chosen_rewards, rejected_rewards)
        accuracy = np.mean(predictions[0] > predictions[1])
        return {"preference_accuracy": accuracy}

    # --- 3. Configure LoRA and RewardTrainer ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    training_args = TrainingArguments(
        output_dir=config.RM_ADAPTER_PATH,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Conservative for pairs
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch=8
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="preference_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        push_to_hub=False
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        loss_type="sigmoid"  # Better for preference ranking
    )

    # --- 4. Train and Save Adapter ---
    trainer.train()
    trainer.save_model()
    print(f"--- ✅ Reward Modeling Stage Complete. Best model adapter saved to {config.RM_ADAPTER_PATH} ---")
    
# =====================================================================================
# STAGE 3: REINFORCEMENT LEARNING (PPO)
# =====================================================================================
def run_ppo():
    """
    PURPOSE: Use the reward model to refine the SFT model via reinforcement learning.
    """
    print("\n--- 🚀 STAGE 3: REINFORCEMENT LEARNING (PPO) ---")

    # --- 1. PPO Configuration ---
    ppo_config = PPOConfig(
        batch_size=4,  # Small for stability
        learning_rate=1.41e-5,
        log_with="none"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- 2. Load Models ---
    # A) Policy Model: Load SFT LoRA, then add value head
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto", 
            token=hf_token
        )
        sft_model = PeftModel.from_pretrained(base_model, config.SFT_ADAPTER_PATH)
        model_with_value_head = AutoModelForCausalLMWithValueHead(sft_model)
        # Re-apply PEFT if needed (value head is on top)
        policy_model = get_peft_model(model_with_value_head, LoraConfig.from_pretrained(config.SFT_ADAPTER_PATH))
    except Exception as e:
        print(f"Policy model loading failed: {e}. Ensure SFT path exists.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # B) Reward Model
    rm_base_model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID, 
        num_labels=1,
        quantization_config=quant_config,
        device_map="auto", 
        token=hf_token
    )
    rm_model = PeftModel.from_pretrained(rm_base_model, config.RM_ADAPTER_PATH)
    rm_model.eval()
    rm_model.config.pad_token_id = tokenizer.pad_token_id

    # Create reference model for KL penalty
    ref_model = create_reference_model(policy_model)

    # --- 3. Initialize PPOTrainer ---
    dataset = load_dataset("json", data_files=config.RM_PPO_DATASET_PATH, split="train[:5%]")  # Smaller for dev

    def format_prompt(example):
        prompt = f"Provide a sentiment analysis summary for the following review:\n\nReview:\n\"{example['review']}\""
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False,
            add_generation_prompt=True
        )
        example['query'] = chat_prompt
        return example  # Tokenize in collator for batching

    formatted_dataset = dataset.map(format_prompt)
    formatted_dataset.set_format(type="torch", columns=["input_ids"])  # Wait, no—PPO needs strings for query

    def collator(data):
        # Tokenize queries on-the-fly for flexibility
        queries = [d["query"] for d in data]
        input_ids = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_SEQ_LENGTH).input_ids
        return {"input_ids": input_ids}
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=formatted_dataset,
        data_collator=collator
    )

    # --- 4. PPO Training Loop ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128
    }

    output_min_length = 10  # Min response length
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=50):  # Limit to 50 batches
        if epoch >= 50: 
            break
        
        query_tensors = batch["input_ids"].to(ppo_trainer.accelerator.device)

        # Generate response from the policy model
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False, 
            output_min_length=output_min_length,
            **generation_kwargs
        )
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Get reward scores from RM (batch properly)
        queries = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        full_texts = [q + r for q, r in zip(queries, batch["response"])]
        
        tokenized_rewards = tokenizer(
            full_texts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        ).to(ppo_trainer.accelerator.device)
        
        with torch.no_grad():
            reward_outputs = rm_model(**tokenized_rewards)
            rewards = reward_outputs.logits.squeeze(-1).cpu().tolist()  # Flat list of scalars
        
        # PPO optimization step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # --- 5. Save the Final Model ---
    policy_model.save_pretrained(config.PPO_MODEL_PATH)
    print(f"--- ✅ PPO Stage Complete. Final model saved to {config.PPO_MODEL_PATH} ---")

if __name__ == '__main__':
    # 🚀 Execute the full RLHF pipeline
    
    # Stage 1: Supervised Fine-Tuning
    run_sft()
    
    # Stage 2: Reward Modeling
    run_reward_modeling()
    
    # Stage 3: Proximal Policy Optimization
    run_ppo()

    print("\n🎉🎉🎉 RLHF Pipeline Complete! 🎉🎉🎉 Test your PPO model now!")