import numpy as np


# 1. CHUNKING
def chunking(documents, chunk_len, overlap_len):
    chunks = []
    chunk_id = 0
    step = chunk_len - overlap_len

    if chunk_len <= 0 or overlap_len < 0 or overlap_len >= chunk_len:
        raise ValueError("Invalid chunk_len or overlap_len")

    for doc_id, document in enumerate(documents):
        words = document.split()

        for start in range(0, len(words), step):
            chunk_words = words[start: start + chunk_len]

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": " ".join(chunk_words)
            })

            chunk_id += 1

            if start + chunk_len >= len(words):
                break

    return chunks


# 2. EMBEDDING
# Assume provided by interviewer.
def embed(texts: list[str]) -> np.ndarray:
    "Returns an array of shape (len(texts), embedding_dim)"

def llm_generate(prompt: str) -> str:
    "Returns the LLM's generated answer."


# 3. BUILD VECTOR STORE
def build_vector_store(chunks):
    
    texts = [chunk["text"] for chunk in chunks]

    embeddings = np.asarray(embed(texts), dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Zero embedding encountered")

    # Normalize embeddings for cosine similarity.
    embeddings = embeddings / norms

    return embeddings, chunks


# 4. RETRIEVAL
def retrieve(query, embeddings, chunks, k=3):
    if k <= 0:
        raise ValueError("k must be positive")

    query_embedding = np.asarray(
        embed([query]), dtype=np.float32
    )

    norm = np.linalg.norm(query_embedding)

    query_embedding = query_embedding[0] / norm

    # Cosine similarity against every chunk.
    scores = embeddings @ query_embedding

    # Get top-k indices in descending similarity order.
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:min(k, len(chunks))]

    results = []

    for idx in top_indices:
        results.append({
            **chunks[idx],
            "score": float(scores[idx])
        })

    return results


# 5. PROMPT CONSTRUCTION
def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    return f"""
Answer the question using the provided context.
If the context does not contain the answer, say you do not know.
Do not invent facts.

Context:
{context}

Question:
{query}

Answer:
"""


# 6. COMPLETE RAG PIPELINE
def rag_pipeline(documents, query, chunk_len=100,
                 overlap_len=20, k=3):

    # Indexing phase
    chunks = chunking(documents, chunk_len, overlap_len)
    embeddings, chunks = build_vector_store(chunks)

    # Retrieval phase
    retrieved_chunks = retrieve(query, embeddings, chunks, k)

    # Generation phase
    prompt = build_prompt(query, retrieved_chunks)
    answer = llm_generate(prompt)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks
    }