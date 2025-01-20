import os
import json
import requests
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, Body
import uvicorn
import redis

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

# Config Constants
LLM_MODEL = "mistral:7b"
EMBED_MODEL = "mistral:7b"
OLLAMA_API_URL = "http://localhost:11434/api"
BATCH_SIZE = 128
CHUNK_SIZE = 1024
EMBED_DIM = 4096  # Mistral typically yields 4096-d embeddings in Ollama

REDIS_HOST = "localhost"
REDIS_PORT = 6379
MAX_QUERY_CACHE = 1000
CACHE_SIM_THRESHOLD = 0.96  # Cosine similarity threshold for cache lookup

PMC_DIR = "./PMC"

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


# Redis Cache Functions
def store_query_in_redis(query_emb: np.ndarray, response: str):
    # Store the query embedding + response in Redis as JSON
    emb_list = query_emb.tolist()[0]
    entry = {"embedding": emb_list, "response": response}
    redis_client.lpush("query_cache", json.dumps(entry))
    # Trim to MAX_QUERY_CACHE to maintain only a limited number of entries in the cache
    redis_client.ltrim("query_cache", 0, MAX_QUERY_CACHE - 1)


def find_similar_query_in_redis(query_emb: np.ndarray) -> Optional[str]:
    """
    Search the Redis list for an embedding whose similarity to `query_emb`
    is >= CACHE_SIM_THRESHOLD. If found, return the stored response. Else None.
    """
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
    # a, b are 1-D float arrays
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# Ollama Synchronous Embedding Calls
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Calls Ollama /api/embeddings for each text individually. Because Mistral in
    Ollama doesn’t batch multiple texts at once, we loop.
    """
    if not texts:
        return np.array([])

    all_embeddings = []
    for text in texts:
        payload = {"model": EMBED_MODEL, "prompt": text}
        try:
            resp = requests.post(
                f"{OLLAMA_API_URL}/embeddings",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding_vec = data.get("embedding", [])
            # If the model fails or returns nothing, skip the round
            if not embedding_vec:
                continue

            all_embeddings.append(embedding_vec)
        except requests.RequestException as e:
            print(f"[Embedding ERROR] {e}")
            return np.array([])

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    if not query.strip():
        return np.array([])

    arr = embed_texts([query])
    if arr.size == 0:
        return np.array([])
    return arr


# Ollama-based LLM Calls (Mistral 7B)
def call_llm(prompt: str) -> str:
    """
    Synchronous call to Mistral 7B via Ollama /api/generate.
    """
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "[LLM ERROR] No response").strip()
    except requests.RequestException as e:
        return f"[LLM ERROR] {e}"


# OpenSearch Setup with HNSW ANN
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = "medical-search-index"

# Attempt to connect to OpenSearch instance
os_client: Optional[OpenSearch] = None
try:
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
    # Check if cluster is up
    info = os_client.info()
    print(f"[INFO] Connected to OpenSearch: {info['version']}")

    # Create index if doesn't exist
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
                            "engine": "nmslib",  # nmslib for cosinesimil
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
    # Stores documents with 'knn_vector' field in OpenSearch using approximate search (HNSW)
    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    def has_any_data(self) -> bool:
        # Returns True if index contains any documents else False.
        if not self.client:
            return False
        try:
            resp = self.client.count(index=self.index_name)
            return resp["count"] > 0
        except:
            return False

    def add_embeddings(self, embeddings: np.ndarray, docs: List[Dict[str, str]]):
        if not self.client or embeddings.size == 0:
            print("[OpenSearchIndexer] No embeddings or no OpenSearch client.")
            return

        actions = []
        # Normalize vectors if using cosine similarity
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

        # Send any remaining actions
        if actions:
            self._bulk_index(actions)

    def _bulk_index(self, actions):
        try:
            success, errors = bulk(self.client, actions)
            print(f"[OpenSearchIndexer] Inserted {success} docs, errors={errors}")
        except Exception as e:
            print(f"[OpenSearchIndexer] Bulk indexing error: {e}")

    def search(
        self, query_emb: np.ndarray, k: int = 3
    ) -> List[Tuple[Dict[str, str], float]]:
        if not self.client or query_emb.size == 0:
            return []

        # Normalize query embedding for cosine similarity
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
            print(f"[OpenSearchIndexer] Found {len(results)} docs in search.")
            return results
        except Exception as e:
            print(f"[OpenSearchIndexer] Search error: {e}")
            return []


# Basic Pre-processing
def basic_cleaning(text: str) -> str:
    return text.replace("\n", " ").strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    # Larger chunk size -> fewer chuinks
    # But each chunk is bigger, which can degrade retrieval precision if chunks are too large
    words = text.split()  # words = [word1, word2, ...]
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk.strip())
    return chunks


# RAG model
class RAGModel:
    def __init__(self):
        self.os_indexer: Optional[OpenSearchIndexer] = None
        if os_client:
            self.os_indexer = OpenSearchIndexer(os_client, OPENSEARCH_INDEX_NAME)

    def build_embeddings_from_scratch(self, pmc_dir: str):
        if not self.os_indexer:
            print("[RAGModel] No OpenSearchIndexer => cannot build embeddings.")
            return

        # If the index has data, skip
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
        # Embed the chunk texts
        chunk_texts = [d["text"] for d in all_docs]
        embs = embed_texts(chunk_texts)
        self.os_indexer.add_embeddings(embs, all_docs)
        print("[RAGModel] Finished embedding & indexing data in OpenSearch.")

    def os_search(self, query_emb: np.ndarray, top_k=3):
        if not self.os_indexer:
            return []

        return self.os_indexer.search(query_emb, k=top_k)

    def ask(self, query: str, top_k: int = 3) -> str:
        """
        1. Embed query and check Redis cache to see if we have a "similar enough" query response
        2. If not found, do approximate search in OpenSearch, retrieve relevant documents
        3. Generate final answer with LLM
        4. Store result in Redis cache for future queries
        """
        if not query.strip():
            return "[ERROR] Empty query."

        # Redis cache lookup
        query_emb = embed_query(query)
        cached_resp = find_similar_query_in_redis(query_emb)
        if cached_resp is not None:
            print(
                "[RAGModel] Found a similar query in Redis cache. Returning cached result."
            )
            return cached_resp

        # Not in cache => do approximate search in OpenSearch
        results = self.os_search(query_emb, top_k=top_k)

        # Group chunks by doc_id to pass relevant documents to LLM
        doc_map = {}
        for doc_dict, score in results:
            doc_id = doc_dict["doc_id"]
            text_chunk = doc_dict["text"]
            if doc_id not in doc_map:
                doc_map[doc_id] = text_chunk
            else:
                doc_map[doc_id] += "\n" + text_chunk

        context_text = ""
        for doc_id, doc_content in doc_map.items():
            context_text += f"--- Document: {doc_id} ---\n{doc_content}\n\n"

        final_prompt = (
            f"User query:\n{query}\n\n"
            f"Relevant documents:\n{context_text}\n\n"
            "Please provide the best possible answer using ONLY the info above. "
            "If the info is insufficient, say so."
        )

        answer = call_llm(final_prompt)
        store_query_in_redis(query_emb, answer)  # cache in Redis
        return answer


import asyncio
from contextlib import asynccontextmanager

# FastAPI integrated for REST APIs
app = FastAPI(
    title="RAG with Ollama + OpenSearch + Redis",
    version="2.0.0",
    description="RAG pipeline to answer medical queries using Ollama embeddings, OpenSearch ANN, and Redis cache.",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Properly initialize RAGModel before API starts and
    clean up resources on shutdown.
    """
    print("[Lifespan] Initializing RAGModel...")
    global rag_model
    rag_model = RAGModel()

    # Run embedding sync inside a separate thread pool for non-blocking startup
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, rag_model.build_embeddings_from_scratch, PMC_DIR)

    print("[Lifespan] RAGModel is ready.")
    yield  # Application is running
    print("[Lifespan] Server is shutting down...")


app.router.lifespan_context = lifespan


def get_rag_model() -> RAGModel:
    return app.state.rag_model


@app.post("/search/opensearch")
async def search_opensearch_route(query: str = Body(...), top_k: int = 50):
    """
    API endpoint for OpenSearch ANN search.
    """
    rag_model = get_rag_model()
    loop = asyncio.get_running_loop()

    # Run embedding generation in thread pool to avoid blocking event loop
    q_emb = await loop.run_in_executor(None, embed_query, query)

    results = await loop.run_in_executor(None, rag_model.os_search, q_emb, top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": [
            {"doc_id": r[0]["doc_id"], "text": r[0]["text"], "score": r[1]}
            for r in results
        ],
    }


@app.post("/ask")
async def ask_route(query: str = Body(...), top_k: int = 50):
    """API endpoint for handling user queries.
    Query -> Embedding Model -> Query Embeddings -> Redis lookup (if not found)
    -> OpenSearch ANN -> Large Lang Model -> Redis cache (store op)
    """

    rag_model = get_rag_model()
    loop = asyncio.get_running_loop()

    # fetch the answer (includes Redis lookup, OpenSearch search, LLM call)
    answer = await loop.run_in_executor(None, rag_model.ask, query, top_k)
    return {"query": query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
