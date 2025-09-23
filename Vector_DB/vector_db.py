# Standard libraries
import os           # for environment variables, paths
import faiss        # FAISS library (Facebook AI Similarity Search) for ANN indexing
import numpy as np  # numerical arrays, embeddings storage

# LangChain embeddings wrapper for Hugging Face models
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain FAISS vector store wrapper
from langchain_community.vectorstores import FAISS

# Docstore backend for LangChain FAISS (stores metadata & docs)
from langchain_community.docstore.in_memory import InMemoryDocstore

# Document schema object in LangChain (holds text + metadata)
from langchain.schema import Document

# Hugging Face Hub login (if model requires auth)
from huggingface_hub import login

# Hugging Face Datasets library (for loading dataset from HF hub)
from datasets import load_dataset

# For environment variable handling (reads .env files)
from dotenv import load_dotenv


# -------------------
# 1. Environment setup
# -------------------
load_dotenv()  # load key=value pairs from .env file into environment variables

def get_env_var(name, default=None):
    """Helper function to fetch environment variables safely."""
    val = os.getenv(name)
    if val is None and default is None:
        raise ValueError(f"Environment variable {name} not set")
    return val if val is not None else default

# Hugging Face login (needed if model is gated/private)
hf_token = get_env_var("HF_TOKEN")          # fetch HF_TOKEN from .env or environment
login(hf_token, add_to_git_credential=True) # login to Hugging Face with token


# -------------------
# 2. Load dataset
# -------------------
dataset_name = "ed-donner/pricer-data"  # dataset repo name on Hugging Face Hub
dataset = load_dataset(dataset_name)    # download & load into DatasetDict (train/test splits)

# Function to clean & normalize dataset entries
def clean_example(example):
    text = example.get("text", "")      # get product text
    price = example.get("price", 0.0)   # get product price (default 0.0)

    # If text is empty → put placeholder
    if not text or text.strip() == "":
        text = "N/A\nPrice is $0.00"

    # If text exists but does not include "Price is $" → append normalized price
    elif "Price is $" not in text:
        text = text.strip() + f"\nPrice is ${price:.2f}"

    # Update text in the example
    example["text"] = text
    return example

# Apply cleaning function to every row in dataset
dataset = dataset.map(clean_example)

# Select the training split
train = dataset["train"]


# -------------------
# 3. Embedding model
# -------------------
embedding_model = "nomic-ai/nomic-embed-text-v1"  # best open-source embedding model

# Create Hugging Face embeddings wrapper
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={
        "device": "cuda",          # run on GPU (use "cpu" if no GPU available)
        "trust_remote_code": True  # needed for models with custom code (like Nomic)
    },
    encode_kwargs={"normalize_embeddings": True}  # normalize vectors for cosine similarity
)


# -------------------
# 4. Convert dataset to Documents
# -------------------
docs = []
for i, ex in enumerate(train):
    # Wrap each dataset row into a LangChain Document
    docs.append(
        Document(
            page_content=ex["text"],           # product text (with price)
            metadata={"id": str(i), "price": ex.get("price", 0.0)}  # metadata includes price
        )
    )


# -------------------
# 5. Compute embeddings
# -------------------
doc_texts = [d.page_content for d in docs]         # extract all texts
doc_embeddings = embeddings.embed_documents(doc_texts)  # compute embeddings for each text
dim = len(doc_embeddings[0])                       # embedding vector dimension (e.g., 768)
doc_embeddings = np.array(doc_embeddings).astype("float32")  # convert to NumPy float32


# -------------------
# 6. Build HNSW index
# -------------------
# Create a FAISS HNSW index:
# - dim = embedding dimension
# - 32 = number of neighbors per node (M parameter)
hnsw_index = faiss.IndexHNSWFlat(dim, 32)

# efConstruction: how thoroughly to connect graph during build (higher = better recall)
hnsw_index.hnsw.efConstruction = 200

# efSearch: how many neighbors to explore during search (higher = better recall)
hnsw_index.hnsw.efSearch = 50

# Add all embeddings into the HNSW index
hnsw_index.add(doc_embeddings)


# -------------------
# 7. Wrap into LangChain FAISS
# -------------------
# Wrap FAISS index into LangChain's VectorStore interface
vectorstore = FAISS(
    embedding_function=embeddings,         # embedding model used
    index=hnsw_index,                      # FAISS HNSW index
    docstore=InMemoryDocstore({}),         # in-memory document store for metadata
    index_to_docstore_id={}                # mapping index→doc ID
)

# Insert all documents into docstore
for i, d in enumerate(docs):
    vectorstore.docstore.add({str(i): d})  # add doc with ID
    vectorstore.index_to_docstore_id[i] = str(i)  # link FAISS vector to docstore ID


# -------------------
# 8. Save vector store
# -------------------
save_path = "/home/lisa/Arupreza/ShopAI/product_vector_store"
os.makedirs(save_path, exist_ok=True)  # ensure save directory exists

# Save FAISS index separately
faiss.write_index(hnsw_index, os.path.join(save_path, "hnsw.index"))

# Save LangChain metadata + docstore
vectorstore.save_local(save_path)

print(f"✅ HNSW vector store saved at {save_path}")


# -------------------
# 9. Reload + test query
# -------------------
# Reload stored vectorstore (index + docs)
loaded = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

# Example query
query = "How much does a Delphi fuel pump cost?"
results = loaded.similarity_search(query, k=2)  # search top-2 similar documents

# Print results
print("\nQuery:", query)
for r in results:
    print("Text:", r.page_content[:200], "...")  # preview first 200 chars of text
    print("Metadata:", r.metadata, "\n")         # print metadata (includes price)