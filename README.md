🛍️ ShopAI — Agentic E‑Commerce Intelligence System

ShopAI is an end‑to‑end intelligent e‑commerce assistant that integrates voice, image, and text modalities to provide price estimation, product sentiment analysis, and personalized recommendations. The long‑term vision is to evolve ShopAI into a fully agentic e‑commerce platform capable of dynamic multi‑agent reasoning and customer interaction.

🚀 Current Development Status (Phase 1)

At present, two major subsystems are complete:

📈 Price Prediction Model

Predicts optimal product prices based on textual descriptions.

Models used: fine‑tuned LLaMA 3.1, Qwen 3, GPT‑2, and RoBERTa baselines.

Evaluation metrics: MAE, RMSE, R², RMSLE, and accuracy (hits%).

Visual outputs: Performance plots stored in Performance_Plot/.

🧠 Product Review Sentiment Model (RLHF)

Implements a full RLHF pipeline for nuanced sentiment reasoning over product reviews.

Phases: Data Generation (LangChain + OpenAI) → Supervised Fine‑Tuning (SFT) → Reward Modeling (RM) → Reinforcement Learning (PPO).

Trains a LLaMA‑3‑based sentiment analyzer capable of distinguishing subtle emotional tones in reviews.

Code located in ProductReviewSentiment/.

🧩 Planned Features (Phase 2 – Agentic Expansion)
🔹 Multimodal Interaction

Voice Control (voice → text) – users can describe products verbally.

OCR Image Search (image → text) – extract product details from photos.

Text‑to‑Text Chat – natural language Q&A with product knowledge base.

🔹 Intelligent Services

Price Estimation according to the product description (already implemented).

Product Analysis (Live) integrating sentiment evaluation from customer reviews.

Similar Product Search via semantic and embedding‑based retrieval.

YouTube Review Integration – analyze sentiment from linked product videos.

🔹 Agentic Workflow

A base conversational model engages the customer to understand intent.

The model defines the problem (price inquiry, review analysis, product comparison, etc.).

Based on intent, it invokes the appropriate specialized agent (price estimator, sentiment analyzer, recommender, etc.).
