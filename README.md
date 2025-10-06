# 🛍️ ShopAI — Agentic E-Commerce Intelligence System

**ShopAI** is an end-to-end intelligent e-commerce assistant integrating **voice**, **image**, and **text** modalities to provide product-level **price estimation**, **sentiment analysis**, and **personalized recommendations**.  
The long-term vision is to evolve ShopAI into a **fully agentic e-commerce platform** capable of dynamic multi-agent reasoning, autonomous customer engagement, and adaptive product intelligence.

---

## 🚀 Current Development Status — *Phase 1: Core Intelligence Layer*

Two primary modules have been implemented and validated.

---

### 📈 Price Prediction Model

**Purpose:** Estimate the optimal market price of a product based on its textual description and features.

- **Models:** fine-tuned *LLaMA 3.1*, *Qwen 3*, *GPT-2*, and *RoBERTa* baselines  
- **Evaluation Metrics:** MAE, RMSE, R², RMSLE, and accuracy (hits %)  
- **Visualization:** Model performance plots available in `Performance_Plot/`  
- **Refer to detailed documentation:** [`PricePredictionModelSrc/README.md`](./PricePredictionModelSrc/README.md)

This module forms the **price intelligence backbone** of ShopAI, enabling cost-aware recommendations and market comparison capabilities.

---

### 🧠 Product Review Sentiment Model (RLHF)

**Purpose:** Build a high-fidelity sentiment analysis engine capable of capturing nuanced emotional and contextual patterns in customer reviews.

#### 🔬 Pipeline Architecture
1. **Data Generation:** Prompt-based synthetic review generation via *LangChain + OpenAI*  
2. **Supervised Fine-Tuning (SFT):** Alignment of LLaMA-3 on structured preference data  
3. **Reward Modeling (RM):** Preference scoring model for reward signal learning  
4. **Reinforcement Learning (PPO):** Fine-tuning with policy optimization to enhance reasoning consistency  
5. **Refer to detailed documentation:** [`ProductReviewSentiment/README.md`](./ProductReviewSentiment/README.md)

> 📂 Codebase: `ProductReviewSentiment/`

This RLHF pipeline trains a **LLaMA-3-based sentiment analyzer** capable of distinguishing subtle tone variations such as sarcasm, polarity ambiguity, and brand bias.

---

## 🧩 Planned Features — *Phase 2: Agentic Expansion*

The next phase aims to extend ShopAI into a **multimodal, multi-agent reasoning system** with real-time customer interaction.

---

### 🔹 Multimodal Interaction
- **Voice Control (Voice → Text):** Users can describe or request products verbally.  
- **OCR Image Search (Image → Text):** Extract structured product data from images or screenshots.  
- **Text-to-Text Chat:** Natural language Q&A over the product knowledge base.

---

### 🔹 Intelligent Services
- ✅ **Price Estimation:** *(already implemented)* contextual prediction via product description.  
- 🔄 **Product Analysis (Live):** Integrates live review streams for real-time sentiment tracking.  
- 🔍 **Similar Product Search:** Embedding-based semantic retrieval for item discovery.  
- 🎥 **YouTube Review Integration:** Extract and analyze sentiment from linked product videos.  
- 🤖 **AI-Powered Customer Support (After-Purchase Service):**  
  A post-order conversational agent that:
  - Tracks delivery status and shipment updates.  
  - Handles refund, replacement, and return requests automatically.  
  - Learns from previous customer interactions to personalize support.  
  - Integrates with product sentiment data to anticipate dissatisfaction and provide proactive solutions.

This customer support agent will form the **post-purchase intelligence layer**, closing the feedback loop between customer experience, review sentiment, and product lifecycle management.

---
