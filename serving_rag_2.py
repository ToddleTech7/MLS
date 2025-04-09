import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import threading
import queue
import time

#########################################
# HYPERPARAMETERS FOR BATCHER
#########################################
MAX_BATCH_SIZE = 4       # e.g., process up to 4 requests at once
MAX_WAITING_TIME = 1.0   # seconds to wait before processing fewer than MAX_BATCH_SIZE
devicee = 'cuda'
#########################################
# Setup: Documents, Models, Tokenizers
#########################################
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(devicee)  # or "cpu"

# For demonstration, we’ll show how to do batch generation with an AutoModelForCausalLM
# If you prefer using pipeline(..., device=0), that can work, but you may need a custom approach for parallel sequences.
CHAT_MODEL_NAME = "facebook/opt-125m"
chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL_NAME).to(devicee)  # or "cpu"

# Precompute doc embeddings on GPU (optional)
doc_embeddings = []

# This method must accept a *list* of texts if we want to do batch embedding
def get_embedding_batch(texts: list[str]) -> np.ndarray:
    """Compute average-pooled embeddings for a batch of strings."""
    inputs = embed_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(devicee) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_dim]
    # We average across seq_len dimension
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

def retrieve_top_k(query_embs: np.ndarray, k: int = 2) -> list[list[str]]:
    """
    Given a batch of query embeddings, retrieve top-k docs for each query.
    query_embs: shape [batch_size, hidden_dim]
    Returns: A list of lists, each containing the top k doc strings.
    """
    # doc_embeddings is shape [num_docs, hidden_dim]
    # we compute doc_embeddings @ query_embs.T => shape [num_docs, batch_size]
    global doc_embeddings
    sims = doc_embeddings @ query_embs.T  # (num_docs, batch_size)
    # For each query in the batch:
    results = []
    for i in range(sims.shape[1]):
        # sort in descending order for doc i
        sim_vec = sims[:, i]
        top_k_indices = np.argsort(sim_vec.ravel())[::-1][:k]
        retrieved_docs = [documents[idx] for idx in top_k_indices]
        results.append(retrieved_docs)
    return results

def batch_generation(prompts: list[str], max_new_tokens=50):
    """
    Example method for parallel generation (batch style).
    We'll use chat_model.generate() and pass input_ids for the batch of prompts.
    """
    # Tokenize prompts in batch
    inputs = chat_tokenizer(prompts, return_tensors="pt", padding=True).to(devicee)
    with torch.no_grad():
        outputs = chat_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # or True, up to you
        )
    # Decode each sequence
    decoded = [chat_tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
    return decoded

def rag_pipeline_batch(queries: list[str], k: int = 2) -> list[str]:
    """
    Batching version of RAG pipeline for multiple queries at once.
    1. embed queries in one shot
    2. retrieve top-k docs per query
    3. generate in batch
    """
    # Step 1: embedding all queries
    query_embs = get_embedding_batch(queries)   # shape [batch_size, hidden_dim]

    # Step 2: retrieval
    retrieved_docs_lists = retrieve_top_k(query_embs, k)  # list of lists of docs

    # Build prompts in a batch
    prompts = []
    for query, docs in zip(queries, retrieved_docs_lists):
        context_str = "\n".join(docs)
        prompt = f"Question: {query}\nContext:\n{context_str}\nAnswer:"
        prompts.append(prompt)

    # Step 3: LLM generation in batch
    # We generate for the entire batch in one call
    generated_texts = batch_generation(prompts, max_new_tokens=50)

    # You might want to parse out only the answer portion. Right now, we decode the full text.
    return generated_texts

#########################################
# FASTAPI + BATCHING IMPLEMENTATION
#########################################
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    k: int = 2

# We'll store requests in a queue
# Each item is a tuple of (QueryRequest, Future-like object)
from concurrent.futures import Future

request_queue = queue.Queue()

# Background worker thread
def batch_worker():
    while True:
        # 1. Collect up to MAX_BATCH_SIZE requests
        #    We'll block for up to MAX_WAITING_TIME to gather a batch
        batch_items = []
        start_time = time.time()

        # Always take at least 1 request
        first_item = request_queue.get()  # blocking wait
        batch_items.append(first_item)

        # We'll keep pulling from the queue until we either:
        # - hit MAX_BATCH_SIZE, or
        # - exceed MAX_WAITING_TIME
        while len(batch_items) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_time
            if elapsed > MAX_WAITING_TIME:
                break
            try:
                item = request_queue.get(timeout=MAX_WAITING_TIME - elapsed)
                batch_items.append(item)
            except queue.Empty:
                break

        # Now we have a batch to process
        # Unpack queries
        queries = [item[0].query for item in batch_items]
        ks = [item[0].k for item in batch_items]

        # For simplicity, we can do a single k for the entire batch
        # or if you want to handle different k's, you'd need a bit more logic
        # We'll just pick the first K or the max?
        k_for_batch = ks[0]

        # RAG in batch
        # If each request can have different k, you'd do it differently
        results = rag_pipeline_batch(queries, k=k_for_batch)

        # Put results back into each Future
        for item, output in zip(batch_items, results):
            future_obj = item[1]
            future_obj.set_result(output)

# Start the worker thread
threading.Thread(target=batch_worker, daemon=True).start()

#########################################
# Initialize doc embeddings just once
#########################################
def init_doc_embeddings():
    global doc_embeddings
    doc_embeddings = get_embedding_batch(documents)  # shape [num_docs, hidden_dim]

init_doc_embeddings()

#########################################
# Inference Endpoint
#########################################
@app.post("/rag")
def predict(payload: QueryRequest):
    # 1. Create a Future
    fut = Future()

    # 2. Put request + future in the queue
    request_queue.put((payload, fut))

    # 3. Wait for the future result
    result = fut.result()  # This will block until batch_worker sets the result
    return {
        "query": payload.query,
        "result": result
    }

#########################################
# Main
#########################################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
