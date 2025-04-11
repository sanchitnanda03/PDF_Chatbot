import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

embedder = SentenceTransformer("intfloat/e5-small-v2")

def create_faiss_index(chunks):
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, chunks, vectors

def get_top_k_chunks(query, index, chunks, vectors, k=3):
    query_vec = embedder.encode([query])
    _, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]

def generate_response_with_llama3(context, query, hf_token):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300}
    }
    response = requests.post(
        "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        headers=headers,
        json=payload
    )
    result = response.json()
    return result[0]["generated_text"].split("Answer:")[-1].strip()
