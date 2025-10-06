from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset, DatasetDict

# ---------------------------
# Tokenizer
# ---------------------------
tok = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or "<pad>"

_preview_done = False

# ---------------------------
# Preprocess function
# ---------------------------
def preprocess_function(
    examples,
    text_column=None,
    label_column="price",
    max_length=256,
    debug=False,
    preview_samples=5,
):
    global _preview_done

    if text_column is None:
        for k, v in examples.items():
            if isinstance(v[0], str):
                text_column = k
                break
        if text_column is None:
            raise ValueError("No string column found for text input.")

    texts = examples[text_column]
    encodings = tok(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    prices = examples[label_column]
    labels = [np.log1p(float(p)) for p in prices]

    if debug and not _preview_done:
        print("\n🔹 Showing preview (first 5 samples only):")
        for i in range(min(preview_samples, len(texts))):
            print(f"\nSample {i+1}:")
            print(f"  Text: {texts[i][:100]} ...")
            print(f"  Price: {prices[i]} -> log1p={labels[i]:.4f}")
            print(f"  input_ids: {encodings['input_ids'][i][:10]} ...")
            print(f"  attention_mask: {encodings['attention_mask'][i][:10]} ...")
        _preview_done = True

    encodings["labels"] = labels
    return encodings


# ---------------------------
# Dataset loader + preprocessing
# ---------------------------
def load_and_preprocess_dataset(
    source: str = "ed-donner/pricer-data",
    text_column: str = "text",
    label_column: str = "price",
    max_length: int = 256,
    seed: int = 42,
):
    ds = load_dataset(source)

    # add validation split if missing
    if "validation" not in ds:
        train_val = ds["train"].train_test_split(test_size=0.1, seed=seed)
        ds = DatasetDict({
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": ds["test"],
        })

    # preprocess each split
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