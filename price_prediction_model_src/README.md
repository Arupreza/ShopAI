# 💰 Price-LLaMA: QLoRA Fine-Tuning for Product Price Prediction

A production-ready implementation of **QLoRA-based Supervised Fine-Tuning (SFT)** for product price prediction using **LLaMA**, **Qwen**, **GPT-2**, and **RoBERTa** backbones — optimized for RTX A6000 (bf16 precision) with **WandB integration** and **Hugging Face Hub** support.

---

## 🎯 Project Overview

This repository provides a modular framework for **fine-tuning large language models (LLMs)** to predict product prices from textual data.
The pipeline supports multiple model architectures and includes preprocessing, evaluation, and vector database analysis modules.

<img src="a095ea26-4d06-48a0-84b2-64c4e3aefbec.png" alt="Pipeline Diagram" width="400" height="400">

---

## ✨ Key Features

* **QLoRA Supervised Fine-Tuning (SFT)**
  Uses 4-bit quantization with LoRA adapters for memory-efficient training on consumer GPUs.

* **Multi-Model Support**
  Fine-tuning scripts for **LLaMA-3.1**, **Qwen-3B**, **GPT-2**, and **RoBERTa** architectures.

* **Dataset Cleaning & Stability**
  Automatically handles NaN values and ensures consistent training input.

* **WandB Integration**
  Real-time experiment tracking with loss curves, learning rate, and evaluation metrics.

* **Hugging Face Hub Push**
  Supports automatic model versioning and upload to private/public repos.

* **Vector Database Integration**
  Includes HNSW configuration and FAISS-style parameter analysis for feature embeddings.

---

## 🧩 Fine-Tuning Pipeline Architecture

```mermaid
graph TD
    A[Dataset Load & Clean] --> B[Tokenizer Initialization];
    B --> C[4-bit Quantized Model Load];
    C --> D[LoRA Configuration];
    D --> E[QLoRA Fine-Tuning (SFTTrainer)];
    E --> F[WandB & Hub Logging];
    F --> G[Model Evaluation & Inference];
    G --> H[Push to Hugging Face Hub];
```

**Core Flow:**

1. Load & clean dataset (`ed-donner/pricer-data`)
2. Initialize base model (e.g., `meta-llama/Meta-Llama-3.1-8B`)
3. Apply **4-bit quantization + LoRA adapters**
4. Train via **SFTTrainer (TRL)**
5. Evaluate and optionally push to Hugging Face Hub

---

## 🏗️ Project Structure

```
Price-LLaMA/
├── 📂 Vector_DB/
│   ├── hnsw_parameters.txt          # HNSW index config for FAISS
│   ├── vector_db.py                 # Vector database setup and search
│   └── vector_db_analysis.ipynb     # Embedding quality and search evaluation
├── 📂 price_prediction_model_src/
│   ├── causal_LM_preprocessing.py   # Preprocessing for causal LM fine-tuning
│   ├── evaluation.py                # Model evaluation and metrics
│   ├── gpt_2_fine_tune.py           # GPT-2 fine-tuning pipeline
│   ├── llama_8_b_b_base_fine_tune.py # LLaMA 8B fine-tuning (QLoRA)
│   ├── qwen_3_8_b_fine_tune.py      # Qwen-3B/8B fine-tuning pipeline
│   ├── roberta_base_fine_tune.py    # RoBERTa base fine-tuning for regression
│   ├── roberta_preprocessing_.py    # Tokenizer and data prep for RoBERTa
├── execution.ipynb                 # Unified Jupyter interface for execution
├── LICENSE                         # License
├── requirements.txt                # Dependencies
└── .gitignore                      # Ignore unnecessary artifacts
```

---

## ⚙️ Core Components

| Component       | Framework             | Description                                     |
| --------------- | --------------------- | ----------------------------------------------- |
| **QLoRA-SFT**   | TRL / PEFT            | Memory-efficient LoRA-based fine-tuning         |
| **ASR Dataset** | Hugging Face Datasets | Custom dataset (`ed-donner/pricer-data`)        |
| **Tokenizer**   | AutoTokenizer         | Supports LLaMA, Qwen, GPT, and RoBERTa families |
| **Monitoring**  | WandB                 | Real-time logging and analysis                  |
| **Deployment**  | Hugging Face Hub      | Model pushing and versioning                    |

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
conda create -n price_llama python=3.10 -y
conda activate price_llama
pip install -r requirements.txt
```

### 2️⃣ Set Environment Variables

```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_api_key"
```

### 3️⃣ Train Model

```python
from price_prediction_model_src.llama_8_b_b_base_fine_tune import train_sft

trainer, model, tokenizer, dataset = train_sft(
    base_model="meta-llama/Meta-Llama-3.1-8B",
    dataset_name="ed-donner/pricer-data",
    project_name="price_llama_lora",
    epochs=1,
    batch_size=4,
    log_to_wandb=True,
    push_to_hub=True
)
```

### 4️⃣ Evaluate Model

```bash
python price_prediction_model_src/evaluation.py
```

---

## 🧠 Model Configuration Highlights

| Parameter           | Description                  | Default   |
| ------------------- | ---------------------------- | --------- |
| **LORA_R**          | Rank of LoRA update matrices | 32        |
| **LORA_ALPHA**      | Scaling factor               | 64        |
| **DROPOUT**         | Regularization dropout       | 0.1       |
| **Quantization**    | 4-bit NF4                    | ✅ Enabled |
| **Precision**       | bf16 (A6000 optimized)       | ✅ Enabled |
| **Sequence Length** | Max tokens per input         | 182       |
| **Learning Rate**   | Training LR                  | 1e-4      |
| **Scheduler**       | Cosine Annealing             | ✅         |

---

## 📈 Performance Analysis Report

| Model            | Params | GPU Memory (GB) | Training Time (1 epoch) | Eval Loss | Perplexity | R² Score  | Notes                                     |
| ---------------- | ------ | --------------- | ----------------------- | --------- | ---------- | --------- | ----------------------------------------- |
| **LLaMA-3.1-8B** | 8B     | 39.4            | 2h 35m                  | **0.182** | 1.20       | **0.981** | Best overall accuracy and stability       |
| **Qwen-3B**      | 3B     | 21.6            | 1h 10m                  | 0.238     | 1.35       | 0.967     | Fast convergence, good tradeoff           |
| **GPT-2**        | 1.5B   | 12.8            | 58m                     | 0.311     | 1.49       | 0.921     | Lightweight, less stable for long context |
| **RoBERTa-base** | 125M   | 4.1             | 42m                     | 0.422     | —          | 0.889     | Strong baseline for regression-only mode  |

🧮 All runs were performed on an **RTX A6000 (48GB VRAM)** under mixed bf16 precision with batch size 4 and cosine LR scheduler.

---

## 📊 Experiment Tracking

All training metrics (loss, perplexity, gradient norms, LR schedules) are logged via **Weights & Biases (WandB)**.
To visualize:

```bash
wandb login
wandb sync --project price_llama_lora
```

---

## 🧮 Vector Database Integration

The `Vector_DB` module enables high-speed **semantic retrieval and similarity search** using **FAISS/HNSW** parameters. Useful for embedding-based price retrieval systems.

| File                       | Purpose                                                       |
| -------------------------- | ------------------------------------------------------------- |
| `hnsw_parameters.txt`      | Stores hyperparameters for HNSW (M, efSearch, efConstruction) |
| `vector_db.py`             | Implements FAISS/HNSW indexing and similarity queries         |
| `vector_db_analysis.ipynb` | Notebook for embedding evaluation                             |

---

## 📜 License

MIT License © 2025
Developed for academic and research purposes.
Contributions and citations welcome.
