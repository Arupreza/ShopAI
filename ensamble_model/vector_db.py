import os
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

# -------------------
# 1. Environment setup
# -------------------
load_dotenv()

def get_env_var(name, default=None):
    val = os.getenv(name)
    if val is None and default is None:
        raise ValueError(f"Environment variable {name} not set")
    return val if val is not None else default

# Hugging Face login (needed if model is gated/private)
hf_token = get_env_var("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

# -------------------
# 2. Load dataset
# -------------------
dataset_name = "ed-donner/pricer-data"
dataset = load_dataset(dataset_name)

def clean_example(example):
    text = example.get("text", "")
    price = example.get("price", 0.0)

    if not text or text.strip() == "":
        text = "N/A\nPrice is $0.00"
    elif "Price is $" not in text:
        text = text.strip() + f"\nPrice is ${price:.2f}"

    example["text"] = text
    return example

dataset = dataset.map(clean_example)
train = dataset["train"]

# -------------------
# 3. Embedding model
# -------------------
embedding_model = "nomic-ai/nomic-embed-text-v1"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={
        "device": "cuda",          # "cpu" if no GPU
        "trust_remote_code": True  # required for nomic model
    },
    encode_kwargs={"normalize_embeddings": True}
)

# -------------------
# 4. Convert dataset to Documents
# -------------------
docs = []
for i, ex in enumerate(train):
    docs.append(
        Document(
            page_content=ex["text"],
            metadata={"id": str(i), "price": ex.get("price", 0.0)}
        )
    )

# -------------------
# 5. Compute embeddings
# -------------------
doc_texts = [d.page_content for d in docs]
doc_embeddings = embeddings.embed_documents(doc_texts)
dim = len(doc_embeddings[0])
doc_embeddings = np.array(doc_embeddings).astype("float32")

# -------------------
# 6. Build HNSW index
# -------------------
hnsw_index = faiss.IndexHNSWFlat(dim, 32)  # 32 = max neighbors per node
hnsw_index.hnsw.efConstruction = 200       # build quality
hnsw_index.hnsw.efSearch = 50              # query quality
hnsw_index.add(doc_embeddings)

# -------------------
# 7. Wrap into LangChain FAISS
# -------------------
vectorstore = FAISS(
    embedding_function=embeddings,
    index=hnsw_index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

# Insert docs into docstore
for i, d in enumerate(docs):
    vectorstore.docstore.add({str(i): d})
    vectorstore.index_to_docstore_id[i] = str(i)

# -------------------
# 8. Save vector store
# -------------------
save_path = "/home/lisa/Arupreza/ShopAI/product_vector_store"
os.makedirs(save_path, exist_ok=True)

faiss.write_index(hnsw_index, os.path.join(save_path, "hnsw.index"))
vectorstore.save_local(save_path)

print(f"✅ HNSW vector store saved at {save_path}")

# -------------------
# 9. Reload + test query
# -------------------
loaded = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

query = "How much does a Delphi fuel pump cost?"
results = loaded.similarity_search(query, k=2)

print("\nQuery:", query)
for r in results:
    print("Text:", r.page_content[:200], "...")
    print("Metadata:", r.metadata, "\n")