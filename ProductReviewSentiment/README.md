# 🧠 ProductReviewSentiment — End-to-End RLHF Pipeline for Sentiment Analysis

This repository provides a **complete RLHF (Reinforcement Learning from Human Feedback)** pipeline for **product review sentiment modeling**, combining **LangChain**, **OpenAI**, and **Hugging Face TRL** to build, fine-tune, and optimize an instruction-following model that produces nuanced and context-aware sentiment analyses.

---

## 📁 Repository Structure

```
ProductReviewSentiment/
├── for_rlhf_data_gen.py                  # Generates preference data using LangChain + OpenAI
├── RLHF_data_for_sentiment_product_review.json  # Auto-generated dataset (chosen/rejected pairs)
└── rlhf.py                               # Full RLHF training pipeline (SFT → RM → PPO)
```

---

## ⚙️ Environment Setup

### 1️⃣ Create Environment

```bash
conda create -n rlhf_sentiment python=3.10 -y
conda activate rlhf_sentiment
```

### 2️⃣ Install Dependencies

```bash
pip install torch transformers accelerate bitsandbytes datasets peft trl \
             langchain langchain-openai pydantic tqdm python-dotenv
```

### 3️⃣ Add API Tokens

Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=sk-your-openai-key
HF_TOKEN=hf_your-huggingface-token
```

---

## 🧩 Step 1 — Preference Data Generation (LangChain)

**Script:** `for_rlhf_data_gen.py`
**Goal:** Generate a high-quality dataset for reward modeling using LangChain and the OpenAI API.

### 🔹 Run:

```bash
python for_rlhf_data_gen.py
```

This script:

* Loads the **`amazon_polarity`** dataset from Hugging Face.
* Uses `ChatOpenAI` with structured schema enforcement (`PydanticOutputParser`).
* Creates preference pairs (`chosen`, `rejected`) like:

```json
{
    "review": "This laptop runs fast but heats up quickly.",
    "chosen": "Mixed sentiment — fast performance but overheating issues.",
    "rejected": "Positive review mentioning fast performance."
}
```

### 💾 Output:

```
RLHF_data_for_sentiment_product_review.json
```

---

## 🧠 Step 2 — Supervised Fine-Tuning (SFT)

**Script:** `rlhf.py`

The first stage of `rlhf.py` fine-tunes **LLaMA-3.1-8B-Instruct** on Amazon product reviews to classify sentiments.
It uses **LoRA adapters** + **4-bit quantization** for efficient training.

### 🔹 Run:

```bash
python rlhf.py
```

**Inside the script:**

* Loads ~1% of `amazon_polarity` for dev-scale training.
* Converts reviews into structured chat format:

  ```
  User: Analyze the sentiment of this review.
  Assistant: Sentiment: Positive
  ```
* Trains LoRA adapter with `SFTTrainer`.

**Saved model:**

```
./sft_llama3_adapters/
```

---

## 🏆 Step 3 — Reward Modeling (RM)

The second stage of `rlhf.py` uses the generated JSON dataset to train a **reward model** that predicts which response (“chosen” vs “rejected”) is better.

**Core idea:** Learn a scalar reward signal where
`R(chosen) > R(rejected)`

### ⚙️ Details:

* Model: `AutoModelForSequenceClassification` (num_labels=1)
* Framework: `TRL RewardTrainer`
* Loss: Sigmoid ranking loss
* Output: Reward adapter saved to:

  ```
  ./rm_llama3_adapters/
  ```

---

## 🤖 Step 4 — Reinforcement Learning (PPO)

Finally, PPO fine-tunes the policy model using the reward model’s feedback.

### ⚙️ PPO Flow

1. Load SFT-trained model + reward model.
2. Generate responses for prompts.
3. Compute rewards via RM.
4. Optimize policy using PPO (`PPOTrainer` from TRL).

### 🔹 Key Components:

* `AutoModelForCausalLMWithValueHead` (adds value head)
* `create_reference_model` (for KL penalty)
* 4-bit quantized training for efficiency

**Output:**

```
./ppo_llama3_model/
```

---

## 📊 End-to-End Pipeline Diagram

```mermaid
graph TD
A[Amazon Product Reviews] --> B[LangChain Dataset Generator]
B --> C[Preference JSON (chosen vs rejected)]
C --> D[SFT: Supervised Fine-Tuning (LLaMA-3.1)]
D --> E[Reward Model Training (RM)]
E --> F[PPO Optimization (RL Stage)]
F --> G[Final RLHF Sentiment Model]
```

---

## 🤪 Testing the Final Model

After completing PPO, test your trained model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./ppo_llama3_model")
model = AutoModelForCausalLM.from_pretrained("./ppo_llama3_model")

prompt = "Review: The phone has a sleek design but poor battery life."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Expected Output:**

> “The review expresses mixed sentiment — design praised, but battery criticized.”

---

## 💾 Training Outputs

| Stage | Directory                                     | Description                |
| ----- | --------------------------------------------- | -------------------------- |
| SFT   | `./sft_llama3_adapters/`                      | Fine-tuned LoRA adapter    |
| RM    | `./rm_llama3_adapters/`                       | Reward model adapter       |
| PPO   | `./ppo_llama3_model/`                         | Final RLHF-optimized model |
| Data  | `RLHF_data_for_sentiment_product_review.json` | Preference dataset         |

---

## ⚡ Recommended Hardware

| Component | Recommended                           |
| --------- | ------------------------------------- |
| GPU       | RTX A6000 / A100 / 4090 (≥24 GB VRAM) |
| RAM       | 32 GB+                                |
| Disk      | 50 GB free space                      |
| CUDA      | 12.x or higher                        |

---

## 🧩 Key Libraries

| Library                          | Use                                     |
| -------------------------------- | --------------------------------------- |
| `langchain` + `langchain-openai` | Data generation with schema enforcement |
| `pydantic`                       | Validating structured outputs           |
| `transformers`                   | Model training and tokenization         |
| `trl`                            | SFT, RM, PPO training                   |
| `peft`                           | LoRA adapter fine-tuning                |
| `bitsandbytes`                   | 4-bit quantization                      |
| `datasets`                       | Dataset loading and mapping             |

---

## 🧠 Conceptual Summary

1. **Generate Data** → LangChain + OpenAI produce labeled (`chosen`, `rejected`) pairs.
2. **SFT** → Teach base model sentiment analysis.
3. **Reward Modeling** → Train scalar model to distinguish better outputs.
4. **PPO Reinforcement Learning** → Optimize base model behavior based on reward signals.
5. **Result** → A refined, RLHF-aligned LLM specialized for nuanced product sentiment reasoning.

---

---

## 🏁 Final Notes

* You can modify `num_samples` in `for_rlhf_data_gen.py` for faster prototyping.
* For real-scale runs, ensure sufficient VRAM (use gradient checkpointing or 8-bit quantization if limited).
* Use the final PPO model for downstream tasks like **review summarization**, **sentiment reasoning**, or **fine-grained aspect analysis**.

---