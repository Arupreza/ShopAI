# model_src/causal_LM_preprocessing.py

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import re

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess_function(examples,
                        text_column="text",
                        label_column="price",
                        max_length=256,
                        debug=False,
                        preview_samples=3):
    texts = []
    labels = []

    for desc, price in zip(examples[text_column], examples[label_column]):
        # Label as string (price formatted)
        label_str = f"{float(price):.2f}"

        # Input prompt
        input_text = f"{desc}\nPrice is $"

        texts.append(input_text)
        labels.append(label_str)

    # Tokenize input (keep special tokens)
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True
    )

    # Tokenize label (digits only, no special tokens!)
    label_encodings = tokenizer(
        labels,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=False   # 👈 this prevents <|end_of_text|> flood
    )

    encodings["labels"] = label_encodings["input_ids"]

    if debug:
        for i in range(min(preview_samples, len(texts))):
            print(f"\nSample {i+1}:")
            print(f"  Input: {texts[i]}")
            print(f"  Label: {labels[i]}")
            print(f"  input_ids[:10]: {encodings['input_ids'][i][:10]}")
            print(f"  labels[:10]: {encodings['labels'][i][:10]}")
            print(f"  Decoded label: {tokenizer.decode([t for t in encodings['labels'][i] if t != tokenizer.pad_token_id])}")

    return encodings



def load_and_preprocess_dataset(
    source="ed-donner/pricer-data",
    text_column="text",
    label_column="price",
    max_length=256,
    seed=42,
):
    ds = load_dataset(source)

    if "validation" not in ds:
        train_val = ds["train"].train_test_split(test_size=0.1, seed=seed)
        ds = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": ds["test"],
        })

    ds = DatasetDict({
        k: v.map(
            lambda x: preprocess_function(
                x,
                text_column=text_column,
                label_column=label_column,
                max_length=max_length,
                debug=(k == "train"),
            ),
            batched=True,
            remove_columns=v.column_names,
        )
        for k, v in ds.items()
    })

    ds.set_format(type="torch")
    return ds