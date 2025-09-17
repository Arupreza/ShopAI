import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def roberta_evaluate_model(model_path: str,
                dataset_name: str = "ed-donner/pricer-data",
                text_column: str = "text",
                label_column: str = "price",
                num_samples: int = 200,
                device: str = "cuda"):

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="regression"
    ).to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset(dataset_name)
    test_data = dataset["test"].select(range(min(num_samples, len(dataset["test"]))))

    truths, preds = [], []

    for dp in test_data:
        text = dp[text_column]
        true_price = float(dp[label_column])

        # Encode input
        inputs = tokenizer(text, return_tensors="pt",
                        truncation=True, padding=True,
                        max_length=256).to(device)

        # Predict
        with torch.no_grad():
            output = model(**inputs).logits.squeeze().item()

        # Convert back from log1p scale
        pred_price = np.expm1(output)

        truths.append(true_price)
        preds.append(pred_price)

    # --- Metrics ---
    mae = mean_absolute_error(truths, preds)
    rmse = math.sqrt(mean_squared_error(truths, preds))
    r2 = r2_score(truths, preds)

    # RMSLE (with clipping to avoid log of negatives)
    rmsle = math.sqrt(np.mean([
        (math.log1p(p) - math.log1p(t)) ** 2 for p, t in zip(preds, truths) if p > 0 and t > 0
    ]))

    # Average signed difference (bias)
    avg_diff = np.mean([p - t for p, t in zip(preds, truths)])

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    max_val = max(max(truths), max(preds))
    plt.plot([0, max_val], [0, max_val], color="deepskyblue", lw=2, alpha=0.6)
    plt.scatter(truths, preds, s=12, c="red", alpha=0.7)
    plt.xlabel("Ground Truth Price")
    plt.ylabel("Predicted Price")
    plt.title(f"MAE={mae:.2f} RMSE={rmse:.2f} R²={r2:.3f} RMSLE={rmsle:.2f} AvgDiff={avg_diff:.2f}")
    plt.show()

    return {"mae": mae, "rmse": rmse, "r2": r2, "rmsle": rmsle, "avg_diff": avg_diff}




#!/usr/bin/env python
# -*- coding: utf-8 -*-

def llama_evaluation(model_path: str, test_amount: int = 250):
    
    import os
    import re
    import math
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel
    
    # =======================================================
    # Constants
    # =======================================================
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    DATASET_NAME = "ed-donner/pricer-data"
    QUANT_4_BIT = True
    TOP_K = 3

    # Colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

    # =======================================================
    # Data
    # =======================================================
    dataset = load_dataset(DATASET_NAME)
    test = dataset["test"]

    # =======================================================
    # Quantization
    # =======================================================
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

    # =======================================================
    # Load Model & Tokenizer
    # =======================================================
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto",
    )

    print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")

    # =======================================================
    # Prediction helpers
    # =======================================================
    def extract_price(s: str) -> float:
        if "Price is $" in s:
            contents = s.split("Price is $")[1]
            contents = contents.replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
            return float(match.group()) if match else 0
        return 0

    def improved_model_predict(prompt: str, device="cuda") -> float:
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(inputs.shape, device=device)

        with torch.no_grad():
            outputs = fine_tuned_model(inputs, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :].to("cpu")
            next_token_probs = F.softmax(next_token_logits, dim=-1)

        top_prob, top_token_id = next_token_probs.topk(TOP_K)
        prices, weights = [], []

        for i in range(TOP_K):
            predicted_token = tokenizer.decode(top_token_id[0][i])
            probability = top_prob[0][i]

            try:
                result = float(predicted_token)
            except ValueError:
                result = 0.0

            if result > 0:
                prices.append(result)
                weights.append(probability)

        if not prices:
            return 0.0

        total = sum(weights)
        weighted_prices = [price * weight / total for price, weight in zip(prices, weights)]
        return sum(weighted_prices).item()

    # =======================================================
    # Evaluation
    # =======================================================
    guesses, truths, errors, sles, colors = [], [], [], [], []

    def color_for(error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    for i in range(min(test_amount, len(test))):
        datapoint = test[i]
        guess = improved_model_predict(datapoint["text"])
        truth = datapoint["price"]

        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error**2
        color = color_for(error, truth)

        title = datapoint["text"].split("\n\n")[1][:20] + "..."

        guesses.append(guess)
        truths.append(truth)
        errors.append(error)
        sles.append(sle)
        colors.append(color)

        print(
            f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} "
            f"Truth: ${truth:,.2f} Error: ${error:,.2f} "
            f"SLE: {sle:,.2f} Item: {title}{RESET}"
        )

    avg_error = sum(errors) / len(errors)
    rmsle = math.sqrt(sum(sles) / len(sles))
    hits = sum(1 for c in colors if c == "green") / len(errors) * 100

    # =======================================================
    # Chart
    # =======================================================
    max_val = max(max(truths), max(guesses))
    plt.figure(figsize=(12, 8))
    plt.plot([0, max_val], [0, max_val], color="deepskyblue", lw=2, alpha=0.6)
    plt.scatter(truths, guesses, s=3, c=colors)
    plt.xlabel("Ground Truth")
    plt.ylabel("Model Estimate")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title(f"Error=${avg_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits:.1f}%")
    plt.show()

    return {"avg_error": avg_error, "rmsle": rmsle, "hits_percent": hits}

# Example call
# results = llama_evaluation("price_llama_lora-2025-09-13_05.09.42/checkpoint-20000", test_amount=200)
# print(results)


def qwen_evaluation(model_path: str, test_amount: int = 250):
    import os
    import re
    import math
    import torch
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    # =======================================================
    # Constants
    # =======================================================
    BASE_MODEL = "Qwen/Qwen3-8B"
    DATASET_NAME = "ed-donner/pricer-data"
    QUANT_4_BIT = True

    # Colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

    # =======================================================
    # Data
    # =======================================================
    dataset = load_dataset(DATASET_NAME)
    test = dataset["test"]

    # =======================================================
    # Quantization
    # =======================================================
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ) if QUANT_4_BIT else BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    # =======================================================
    # Load Model & Tokenizer
    # =======================================================
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto",
    )

    print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")

    # =======================================================
    # Prediction helpers
    # =======================================================
    def extract_price(s: str) -> float:
        """Extract the first number after 'Price is $'."""
        if "Price is $" in s:
            contents = s.split("Price is $")[1]
            contents = contents.replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
            return float(match.group()) if match else 0
        return 0.0

    def improved_model_predict(prompt: str, device="cuda") -> float:
        """Generate full continuation after 'Price is $' and extract number."""
        set_seed(42)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = fine_tuned_model.generate(
                **inputs,
                max_new_tokens=10,     # enough to output full number
                do_sample=False,       # greedy decode
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return extract_price(decoded)

    # =======================================================
    # Evaluation
    # =======================================================
    guesses, truths, errors, sles, colors = [], [], [], [], []

    def color_for(error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    for i in range(min(test_amount, len(test))):
        datapoint = test[i]
        guess = improved_model_predict(datapoint["text"])
        truth = datapoint["price"]

        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error**2
        color = color_for(error, truth)

        title = datapoint["text"][:40].replace("\n", " ")

        guesses.append(guess)
        truths.append(truth)
        errors.append(error)
        sles.append(sle)
        colors.append(color)

        print(
            f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} "
            f"Truth: ${truth:,.2f} Error: ${error:,.2f} "
            f"SLE: {sle:,.2f} Item: {title}{RESET}"
        )

    avg_error = sum(errors) / len(errors)
    rmsle = math.sqrt(sum(sles) / len(sles))
    hits = sum(1 for c in colors if c == "green") / len(errors) * 100

    # =======================================================
    # Chart
    # =======================================================
    max_val = max(max(truths), max(guesses))
    plt.figure(figsize=(12, 8))
    plt.plot([0, max_val], [0, max_val], color="deepskyblue", lw=2, alpha=0.6)
    plt.scatter(truths, guesses, s=3, c=colors)
    plt.xlabel("Ground Truth")
    plt.ylabel("Model Estimate")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title(f"Error=${avg_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits:.1f}%")
    plt.show()

    return {"avg_error": avg_error, "rmsle": rmsle, "hits_percent": hits}


# Example usage:
# results = qwen_evaluation("pricer_qwen3_8b-2025-09-14_11.30.00/checkpoint-1000", test_amount=200)