import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import re

DOC_PATTERN = r"(?i)(document(?:\s+id)?\s+['\"]?[A-Za-z0-9._-]+['\"]?)"

from fastapi import FastAPI, Body
import uvicorn
import redis

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

import asyncio
from contextlib import asynccontextmanager

import httpx

# Constants
OLLAMA_API_URL = "http://localhost:11434/api"
EMBED_MODEL_NAME = "jina/jina-embeddings-v2-base-de"
LLM_MODEL_NAME = "mistral:7b"
BATCH_SIZE = 128
CHUNK_SIZE = 512
EMBED_DIM = 768
MAX_EMBED_CONCURRENCY = 5  # Concurrency limit to avoid overwhelming Ollama

REDIS_HOST = "localhost"
REDIS_PORT = 6379
MAX_QUERY_CACHE = 1000
CACHE_SIM_THRESHOLD = 0.96  # Cosine similarity threshold for cache lookup

PMC_DIR = "./PMC"

# ==============================================================================
# Redis Client & Cache Functions
# ==============================================================================
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


def store_query_in_redis(query_emb: np.ndarray, response: str):
    """Store a query embedding and its response in Redis."""
    emb_list = query_emb.tolist()[0]
    entry = {"embedding": emb_list, "response": response}
    redis_client.lpush("query_cache", json.dumps(entry))
    redis_client.ltrim("query_cache", 0, MAX_QUERY_CACHE - 1)


def find_similar_query_in_redis(query_emb: np.ndarray) -> Optional[str]:
    """Search Redis for an embedding similar enough to query_emb and return the cached response if found."""
    cached_list = redis_client.lrange("query_cache", 0, MAX_QUERY_CACHE - 1)
    query_vec = query_emb[0]

    best_sim = -1.0
    best_response = None

    for c in cached_list:
        entry = json.loads(c)
        emb_list = entry["embedding"]
        cached_emb = np.array(emb_list, dtype=np.float32)
        sim = cosine_similarity(query_vec, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_response = entry["response"]

    if best_sim >= CACHE_SIM_THRESHOLD:
        return best_response

    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ==============================================================================
# Asynchronous Ollama API Functions
# ==============================================================================
async def ollama_embed_text(text: str, model: str = EMBED_MODEL_NAME) -> List[float]:
    """
    Request an embedding for `text` from Ollama via HTTP POST.
    Expects JSON of form {"model": model, "prompt": text} and returns {"embedding": [...]}
    """
    async with httpx.AsyncClient() as client:
        payload = {"model": model, "prompt": text, "stream": False}
        resp = await client.post(
            f"{OLLAMA_API_URL}/embeddings", json=payload, timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()
        # print(f"Embedding generated = {data.get('embedding', 'No embedding field')}")
        return data.get("embedding", [])


async def embed_texts_in_batches(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embeds a list of texts in smaller batches, respecting concurrency limits.
    """
    if not texts:
        return np.array([])

    all_embeddings = []
    concurrency_sem = asyncio.Semaphore(MAX_EMBED_CONCURRENCY)

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        async def embed_single(txt: str) -> List[float]:
            async with concurrency_sem:
                return await ollama_embed_text(txt)

        tasks = [embed_single(txt) for txt in batch]
        batch_embeddings = await asyncio.gather(*tasks)
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


async def embed_query(query: str) -> np.ndarray:
    """Obtain an embedding for a single query."""
    if not query.strip():
        return np.array([])

    emb = await ollama_embed_text(query, EMBED_MODEL_NAME)
    return np.array([emb], dtype=np.float32)


async def ollama_generate_text(prompt: str, model: str = LLM_MODEL_NAME) -> str:
    """
    Request text generation from Ollama via HTTP POST.
    Returns the "generated_text" field from the JSON.
    """
    async with httpx.AsyncClient() as client:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. You must follow these rules:\n"
                        "1) Always cite document IDs from the context exactly as 'Document XYZ' without file extension.\n"
                        "2) Do not add your chain-of-thought.\n"
                        "3) You must answer exactly the user query and not the context but only use the context information to find your answer specific info.\n"
                        "4) Keep the answer short and precise like at most 4 sentences.\n"
                        "5) If you lack context, say so.\n\n"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.6,
                # "top_k": 50,
                # "top_p": 0.9,
                "max_tokens": 256,  # Limit the token count so the model does not ramble
            },
        }
        resp = await client.post(f"{OLLAMA_API_URL}/chat", json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        # print(f"Result = {data}")
        return data.get("message", {}).get("content", "").strip()


def _contains_document_id(text: str) -> bool:
    """Helper function to check if the text has any recognized document references."""
    matches = re.findall(DOC_PATTERN, text)
    return len(matches) > 0


async def call_llm(prompt: str) -> str:
    """A wrapper to call the Ollama LLM generation endpoint."""
    # 1) Get the initial response from the LLM
    return await ollama_generate_text(prompt)

    # # 2) Check if the response cites any document ID using regex
    # if _contains_document_id(first_response):
    #     return first_response
    # else:
    #     # 3) Prompt the LLM to revise if no document IDs were found
    #     revised_prompt = (
    #         "You did not cite any Document Name/ID. Please revise your answer. "
    #         "Remember to cite Document Name/ID in the format found in the original context. "
    #         "Use the exact Document ID(s) if available.\n\n"
    #         f"Here is your previous answer:\n{first_response}\n\n"
    #         f"Original Prompt & Context:\n{prompt}\n\n"
    #         "Now, please provide a revised answer ensuring you include Document ID(s)."
    #     )
    #     second_response = await ollama_generate_text(revised_prompt)

    #     # 4) Ask the model to pick the better response (A or B)
    #     compare_prompt = (
    #         "Below are two versions of your response, labeled A and B.\n\n"
    #         f"Prompt: {prompt}\n\n"
    #         f"A: {first_response}\n\n"
    #         f"B: {second_response}\n\n"
    #         "Which version (A or B) better satisfies the requirement "
    #         "to cite the Document Name/ID accurately? Provide only one version as the final answer "
    #         "and keep the references as-is if they match the instructions."
    #     )
    #     final_response = await ollama_generate_text(compare_prompt)

    #     # Optional: Check if the final response now cites any Document ID
    #     # If it still doesn't, you might prompt again or just return as-is.
    #     if not _contains_document_id(final_response):
    #         # (Optional) fallback: e.g., default to second_response or keep final_response
    #         # Based on your needs, you might do one more iteration, or just return the second_response.
    #         # For example:
    #         return second_response

    #     return final_response


# ==============================================================================
# OpenSearch Setup with HNSW ANN
# ==============================================================================
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = "medical-search-index"

os_client: Optional[OpenSearch] = None
try:
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
    info = os_client.info()
    print(f"[INFO] Connected to OpenSearch: {info['version']}")

    if not os_client.indices.exists(OPENSEARCH_INDEX_NAME):
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBED_DIM,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "parameters": {"m": 16, "ef_construction": 200},
                        },
                    },
                }
            },
        }
        resp = os_client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_body)
        print(f"[INFO] Created '{OPENSEARCH_INDEX_NAME}' index with HNSW: {resp}")
    else:
        print(f"[INFO] OpenSearch index '{OPENSEARCH_INDEX_NAME}' is ready.")
except Exception as e:
    print(f"[WARNING] OpenSearch not initialized: {e}")
    os_client = None


class OpenSearchIndexer:
    """
    Index documents with embeddings into OpenSearch using the HNSW algorithm.
    """

    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    def has_any_data(self) -> bool:
        if not self.client:
            return False
        try:
            resp = self.client.count(index=self.index_name)
            return resp["count"] > 0
        except Exception:
            return False

    def add_embeddings(self, embeddings: np.ndarray, docs: List[Dict[str, str]]):
        if not self.client or embeddings.size == 0:
            print("[OpenSearchIndexer] No embeddings or no OpenSearch client.")
            return

        actions = []
        # Normalize each embedding vector for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        for i, (doc_dict, emb) in enumerate(zip(docs, embeddings)):
            doc_id = doc_dict["doc_id"]
            text_content = doc_dict["text"]
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": f"{doc_id}_{i}",
                    "_source": {
                        "doc_id": doc_id,
                        "text": text_content,
                        "embedding": emb.tolist(),
                    },
                }
            )
            if len(actions) >= BATCH_SIZE:
                self._bulk_index(actions)
                actions = []
        if actions:
            self._bulk_index(actions)

    def _bulk_index(self, actions):
        try:
            success, errors = bulk(self.client, actions)
            print(f"[OpenSearchIndexer] Inserted {success} docs, errors={errors}")
        except Exception as e:
            print(f"[OpenSearchIndexer] Bulk indexing error: {e}")

    def search(
        self, query_emb: np.ndarray, k: int = 5
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or query_emb.size == 0:
            return []

        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (q_norm + 1e-9)
        vector = query_emb[0].tolist()
        query_body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
        }
        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            hits = resp["hits"]["hits"]
            results = []
            for h in hits:
                doc_score = h["_score"]
                doc_source = h["_source"]
                results.append((doc_source, float(doc_score)))

            print(
                f"[OpenSearchIndexer] Found {len(results)} relevant results in search."
            )
            return results
        except Exception as e:
            print(f"[OpenSearchIndexer] Search error: {e}")
            return []


# ==============================================================================
# Basic Pre-processing Functions
# ==============================================================================
def basic_cleaning(text: str) -> str:
    return text.replace("\n", " ").strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits text into chunks by word count.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk.strip())
    return chunks


# ==============================================================================
# RAG Model
# ==============================================================================
class RAGModel:
    def __init__(self):
        self.os_indexer: Optional[OpenSearchIndexer] = None
        if os_client:
            self.os_indexer = OpenSearchIndexer(os_client, OPENSEARCH_INDEX_NAME)

    async def build_embeddings_from_scratch(self, pmc_dir: str):
        """
        Reads text files from the given directory, splits them into chunks, obtains embeddings via Ollama,
        and indexes them in OpenSearch.
        """
        if not self.os_indexer:
            print("[RAGModel] No OpenSearchIndexer => cannot build embeddings.")
            return

        if self.os_indexer.has_any_data():
            print("[RAGModel] OpenSearch already has data. Skipping embedding.")
            return

        print("[RAGModel] Building embeddings from scratch...")
        data_files = os.listdir(pmc_dir)
        all_docs = []

        for fname in data_files:
            if fname.startswith("PMC") and fname.endswith(".txt"):
                path = os.path.join(pmc_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="latin-1") as f:
                        text = f.read()

                cleaned_text = basic_cleaning(text)
                text_chunks = chunk_text(cleaned_text, CHUNK_SIZE)
                for chunk_str in text_chunks:
                    all_docs.append({"doc_id": fname, "text": chunk_str})

        if not all_docs:
            print("[RAGModel] No text found in directory. Exiting.")
            return

        print(f"[RAGModel] Generating embeddings for {len(all_docs)} chunks...")
        chunk_texts = [d["text"] for d in all_docs]

        # Use batch-based embedding approach with concurrency control
        embs = await embed_texts_in_batches(chunk_texts, batch_size=32)

        loop = asyncio.get_running_loop()
        # Offload synchronous OpenSearch indexing to a thread pool
        await loop.run_in_executor(None, self.os_indexer.add_embeddings, embs, all_docs)
        print("[RAGModel] Finished embedding & indexing data in OpenSearch.")

    def os_search(self, query_emb: np.ndarray, top_k=5):
        if not self.os_indexer:
            return []

        return self.os_indexer.search(query_emb, k=top_k)

    async def ask(self, query: str, top_k: int = 5) -> str:
        """
        Process a user query:
          1. Get its embedding via Ollama.
          2. Check Redis cache for a similar query.
          3. If not cached, perform a k-NN search in OpenSearch.
          4. Generate an answer via the Ollama LLM.
          5. Cache and return the answer.
        """
        if not query.strip():
            return "[ERROR] Empty query."

        # 1) Embed query
        query_emb = await embed_query(query)

        # 2) Check Redis cache
        cached_resp = find_similar_query_in_redis(query_emb)
        if cached_resp is not None:
            print(
                "[RAGModel] Found a similar query in Redis cache. Returning cached result."
            )
            return cached_resp

        # 3) Search OpenSearch
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.os_search, query_emb, top_k)

        # 4) Prepare context & call LLM
        doc_map = {}
        for doc_dict, score in results:
            doc_id = doc_dict["doc_id"]
            text_chunk = doc_dict["text"]
            if doc_id not in doc_map:
                doc_map[doc_id] = text_chunk
            else:
                doc_map[doc_id] += "\n" + text_chunk

        print(f"[OpenSearchIndexer] Found {len(doc_map)} docs in search.")

        context_text = ""
        for doc_id, doc_content in doc_map.items():
            print(doc_id)
            context_text += f"--- Document: {doc_id} ---\n{doc_content}\n\n"

        # print(context_text)

        final_prompt = (
            "INSTRUCTIONS: You must cite the **Document ID or Name** for any information you use. "
            "Your answer must follow this format: Direct Answer (at most 4 sentences)\n -> References (list any Document IDs from which the answer is extracted, e.g., Document XYZ). "
            "Your answer must be specific to the user query regardless of the provided context information. "
            "If the Document ID is 'PMC555957.txt', refer to it as 'Document PMC555957' (without the file extension). "
            "If multiple documents are relevant, cite each of them explicitly. "
            "Do NOT reveal your reasoning or chain of thought.\n\n"
            f"User Query:\n{query}\n\n"
            "Context:\n"
            f"{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your answer strictly based on the above information. If insufficient or not relevant, then say so."
        )

        # final_prompt = (
        #     f"User query which you got to answer:\n{query}\n\n"
        #     f"Base your answer on the relevant information or context provided in the following documents:\n"
        #     f"{context_text}\n"
        #     "--- End of context ---\n\n"
        #     "Follow the below instructions very carefully and provide the answer:\n"
        #     "Keep the answer specific to the user query and do not return irrelevant information which is not relevant to the user query. "
        #     "You must cite the specific Document ID exactly as in the context provided. "
        #     "For example, if the Document ID is 'PMC555957.txt', refer to it as 'Document PMC555957' in your answer (without the file extension). "
        #     "If multiple documents are relevant, cite them explicitly. "
        #     "Please provide the best possible answer using ONLY the user query given above and the provided context information also given above. "
        #     "Do not mention your chain of thought. "
        #     "If the info is insufficient, say so."
        # )

        answer = await call_llm(final_prompt)

        # 5) Cache in Redis
        store_query_in_redis(query_emb, answer)
        return answer


# ==============================================================================
# FastAPI Application Setup
# ==============================================================================
app = FastAPI(
    title="RAG with Ollama Models + OpenSearch + Redis",
    version="3.0.0",
    description="RAG pipeline to answer queries using Ollama's embedding/LLM, OpenSearch ANN, & Redis cache.",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Initializing RAGModel...")
    global rag_model
    rag_model = RAGModel()
    await rag_model.build_embeddings_from_scratch(PMC_DIR)
    print("[Lifespan] RAGModel is ready.")
    yield
    print("[Lifespan] Server is shutting down...")


app.router.lifespan_context = lifespan


def get_rag_model() -> RAGModel:
    return rag_model


@app.post("/search/opensearch")
async def search_opensearch_route(query: str = Body(...), top_k: int = 5):
    """
    API endpoint for performing approximate k-NN search via OpenSearch.
    """
    rm = get_rag_model()
    if not rm:
        return {"error": "RAGModel not initialized."}

    # Embed query text
    q_emb = await embed_query(query)

    # Offload synchronous OS search to a thread
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, rm.os_search, q_emb, top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": [
            {"doc_id": r[0]["doc_id"], "text": r[0]["text"], "score": r[1]}
            for r in results
        ],
    }


@app.post("/ask")
async def ask_route(query: str = Body(...), top_k: int = 5):
    """
    API endpoint that processes a user query through the entire RAG pipeline.
    """
    rm = get_rag_model()
    if not rm:
        return {"error": "RAG Model not initialized"}

    answer = await rm.ask(query, top_k)
    return {"query": query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
