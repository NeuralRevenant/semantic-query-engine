import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body
import uvicorn

# ------------------------------------------------------------------------------
# Redis
# ------------------------------------------------------------------------------
import redis

# ------------------------------------------------------------------------------
# OpenSearch
# ------------------------------------------------------------------------------
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

# ------------------------------------------------------------------------------
# PostgreSQL
# ------------------------------------------------------------------------------
import psycopg2
from psycopg2.extras import DictCursor

# ------------------------------------------------------------------------------
# Transformers (for embeddings)
# ------------------------------------------------------------------------------
import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------------------------
# spaCy
# ------------------------------------------------------------------------------
import spacy

# ------------------------------------------------------------------------------
# HTTP client for remote LLM calls
# ------------------------------------------------------------------------------
import httpx


# ==============================================================================
# Configuration Constants
# ==============================================================================
# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379

REDIS_MAX_ITEMS = 1000
REDIS_CACHE_LIST = "query_cache_lfu"

# Embeddings
EMBED_DIM = 768
CHUNK_SIZE = 512

# ---- Batching & Concurrency ----
BATCH_SIZE = 64  # For embedding documents in batches
MAX_EMBED_CONCURRENCY = (
    5  # Limit the number of concurrent embedding tasks to avoid GPU overload
)


# Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    inference_dtype = torch.float16  # Faster on most NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    inference_dtype = torch.float32  # float32 better on MPS
else:
    device = torch.device("cpu")
    inference_dtype = torch.float32  # CPU => stick to float32

print(f"device: {device.type}")
print(f"inference data type: {inference_dtype}")

# Model name
EMBED_MODEL_NAME = "jinaai/jina-embeddings-v2-base-de"

# Remote LLM endpoint
REMOTE_LLM_URL = os.getenv("REMOTE_LLM_URL", ".../generate")

# PostgreSQL
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# OpenSearch
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", 9200))
OPENSEARCH_INDEX_NAME = "medical-search-index"

# Local text files
PMC_DIR = ""

# ==============================================================================
# Redis LFU Cache
# ==============================================================================
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def lfu_cache_get(query_emb: np.ndarray) -> Optional[str]:
    """Retrieve from Redis if we find a sufficiently similar embedding."""
    cached_list = redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
    if not cached_list:
        return None

    query_vec = query_emb[0]
    best_sim = -1.0
    best_index = -1
    best_entry_data = None

    for i, item in enumerate(cached_list):
        entry = json.loads(item)
        emb_list = entry["embedding"]

        cached_emb = np.array(emb_list, dtype=np.float32)
        sim = cosine_similarity(query_vec, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_index = i
            best_entry_data = entry

    if best_sim < 0.96:
        return None

    if best_entry_data:
        # increment freq
        best_entry_data["freq"] = best_entry_data.get("freq", 1) + 1
        # update entry
        redis_client.lset(REDIS_CACHE_LIST, best_index, json.dumps(best_entry_data))
        return best_entry_data["response"]

    return None


def _remove_least_frequent_item():
    """Helper to remove the least frequently used entry from Redis."""
    cached_list = redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
    if not cached_list:
        return

    min_freq = float("inf")
    min_index = -1
    for i, item in enumerate(cached_list):
        entry = json.loads(item)
        freq = entry.get("freq", 1)
        if freq < min_freq:
            min_freq = freq
            min_index = i

    if min_index >= 0:
        item_str = cached_list[min_index]
        redis_client.lrem(REDIS_CACHE_LIST, 1, item_str)


def lfu_cache_put(query_emb: np.ndarray, response: str):
    """Insert new entry into the LFU Redis cache."""
    entry = {"embedding": query_emb.tolist()[0], "response": response, "freq": 1}
    current_len = redis_client.llen(REDIS_CACHE_LIST)
    if current_len >= REDIS_MAX_ITEMS:
        _remove_least_frequent_item()

    redis_client.lpush(REDIS_CACHE_LIST, json.dumps(entry))


# ==============================================================================
# PostgreSQL
# ==============================================================================
def get_postgres_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        cursor_factory=DictCursor,
    )


def find_most_recent(limit: int = 5) -> List[Dict[str, Any]]:
    query = f"""
        SELECT id, title, abstract, date
        FROM documents
        ORDER BY date DESC
        LIMIT {limit}
    """
    try:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print("[Postgres] find_most_recent error:", e)
        return []


def find_highly_cited(limit: int = 5) -> List[Dict[str, Any]]:
    query = f"""
        SELECT id, title, abstract, citation_count
        FROM documents
        ORDER BY citation_count DESC
        LIMIT {limit}
    """
    try:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print("[Postgres] find_highly_cited error:", e)
        return []


# ==============================================================================
# spaCy for Intent Classification
# ==============================================================================
print("[Load] Attempting spaCy text-cat model...")
SPACY_MODEL_PATH = "./model_nlu"
spacy_textcat_nlp: Optional[spacy.Language] = None
try:
    spacy_textcat_nlp = spacy.load(SPACY_MODEL_PATH)
    print("[Load] spaCy loaded successfully.")
except Exception as ex:
    print("[Warning] spaCy load error:", ex)
    spacy_textcat_nlp = None


def spacy_classify(query: str) -> Optional[str]:
    if spacy_textcat_nlp is None:
        return None
    doc = spacy_textcat_nlp(query)
    if not doc.cats:
        return None
    return max(doc.cats, key=doc.cats.get)


def fallback_classify(query: str) -> str:
    """Simple fallback heuristics if spaCy classification fails."""
    qlower = query.lower()
    if "recent" in qlower:
        return "find_most_recent"
    elif "cited" in qlower:
        return "find_highly_cited"
    elif "count" in qlower:
        return "count_documents"
    return "retrieve_info"


def combined_intent_classification(query: str) -> str:
    label_spacy = spacy_classify(query)
    if label_spacy:
        return label_spacy
    return fallback_classify(query)


CATEGORY_TOPK_MAP = {
    "count_documents": 100,
    "retrieve_info": 3,
    "find_most_recent": 3,
    "find_highly_cited": 3,
    "other": 3,
}

# ==============================================================================
# Hugging Face Model + Tokenizer (Jina Embeddings)
# ==============================================================================
print("[Load] Initializing Jina Embeddings:", EMBED_MODEL_NAME)
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(
    EMBED_MODEL_NAME,
    torch_dtype=inference_dtype,
    attn_implementation="eager",  # Ensure SDPA is not bypassed
).to(device)

embedding_model.eval()  # Disable dropout for deterministic inference, and so that model doesn't randomly drop neurons

# # Compile model for faster inference (with PyTorch 2.0+ versions) - error with transformers library!
# if torch.__version__ >= "2.0":
#     embedding_model = torch.compile(
#         embedding_model, mode="reduce-overhead"
#     )  # Best for inference


def _attention_mask_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Typical attention-mask mean pooling: sum(token_embeddings * mask) / sum(mask)
    similar to how 'sentence-transformers' style models do mean pooling.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@torch.inference_mode()
def embed_single_text(text: str) -> np.ndarray:
    """Embed a single text with attention-mask mean pooling in half-precision if NVIDIA CUDA GPU, else float32."""
    if not text.strip():
        return np.zeros((EMBED_DIM,), dtype=np.float32)

    # Enable async tensor transfers using non-blocking=true option
    inputs = embedding_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    # Use autocast if we are on CUDA NVIDIA GPU for half-precision, otherwise not supported
    if device.type == "cuda":
        with autocast(device_type="cuda", enabled=True):
            outputs = embedding_model(**inputs, output_attentions=False, head_mask=None)
    else:
        outputs = embedding_model(**inputs, output_attentions=False, head_mask=None)

    hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]
    pooled = _attention_mask_mean_pool(hidden_states, inputs["attention_mask"])

    # Convert to float32 before NumPy conversion to avoid NaNs
    emb_np = pooled[0].detach().cpu().to(torch.float32).numpy()
    # emb_np = pooled[0].detach().cpu().numpy().astype(np.float32)
    return emb_np


def embed_query_jina(query: str) -> np.ndarray:
    """
    Embeds a single query for real-time usage (e.g., user queries).
    """
    emb = embed_single_text(query)
    return np.expand_dims(emb, axis=0)


async def embed_texts_in_batches(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Asynchronously embed a list of texts in small batches, using standard attention-mask mean pooling
    for each batch. We use a semaphore to run up to MAX_EMBED_CONCURRENCY batches in parallel.
    """

    # If no texts, return an empty array
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    # Split the 'texts' into sub-batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch_chunk = texts[i : i + batch_size]
        batches.append(batch_chunk)

    # Create a semaphore to limit concurrency
    concurrency_sem = asyncio.Semaphore(MAX_EMBED_CONCURRENCY)

    async def embed_one_batch(batch_texts: List[str]) -> np.ndarray:
        """
        Embeds a single batch of texts using attention-mask mean pooling.
        We acquire the semaphore before doing any heavy embedding work.
        """
        async with concurrency_sem:
            # Prepare inputs
            inputs = embedding_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.inference_mode():
                if device.type == "cuda":
                    with autocast(device_type="cuda", enabled=True):
                        outputs = embedding_model(
                            **inputs, output_attentions=False, head_mask=None
                        )
                else:
                    outputs = embedding_model(
                        **inputs, output_attentions=False, head_mask=None
                    )

            hidden_states = (
                outputs.last_hidden_state
            )  # shape: [batch_size, seq_len, 768]
            # Weighted mean pooling using attention mask
            # print(f"Outputs: {outputs}")
            pooled = _attention_mask_mean_pool(hidden_states, inputs["attention_mask"])
            # print(f"return value from embed_one_batch: {pooled}")
            # Convert to float32 before NumPy conversion to avoid NaNs
            # return pooled.detach().cpu().numpy().astype(np.float32)
            return pooled.detach().cpu().to(torch.float32).numpy()

    # Schedule embedding tasks for each batch
    tasks = [embed_one_batch(batch) for batch in batches]

    # Gather all batch-embedding results (in parallel, limited by the semaphore)
    results = await asyncio.gather(*tasks)

    # Concatenate all batch embeddings into one array
    return np.concatenate(results, axis=0)  # all_embeddings


# ==============================================================================
# Remote LLM Call
# ==============================================================================
async def call_remote_llm(
    context_text: str, user_query: str, max_tokens: int = 200
) -> str:
    payload = {
        "context": context_text,
        "query": user_query,
        "max_new_tokens": max_tokens,
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(REMOTE_LLM_URL, json=payload, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return data["answer"]
        except Exception as e:
            print("[Remote LLM] Error:", e)
            return "Sorry, the remote LLM is unavailable."


# ==============================================================================
# OpenSearch Setup
# ==============================================================================
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
    print("[OpenSearch] Connected:", info["version"])

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
                            "parameters": {"m": 48, "ef_construction": 400},
                        },
                    },
                }
            },
        }
        resp = os_client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_body)
        print("[OpenSearch] Created index:", resp)
    else:
        print(f"[OpenSearch] Index '{OPENSEARCH_INDEX_NAME}' ready.")
except Exception as e:
    print("[OpenSearch] Initialization error:", e)
    os_client = None


class OpenSearchIndexer:
    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name

    def has_any_data(self) -> bool:
        if not self.client:
            return False
        try:
            c = self.client.count(index=self.index_name)
            return c["count"] > 0
        except:
            return False

    def add_embeddings(self, embeddings: np.ndarray, docs: List[Dict[str, str]]):
        if not self.client or embeddings.size == 0:
            return

        actions = []
        # Normalize for better KNN with cosinesimil
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            doc_id = doc["doc_id"]
            text_content = doc["text"]
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": f"{doc_id}_{i}",
                "_source": {
                    "doc_id": doc_id,
                    "text": text_content,
                    "embedding": emb.tolist(),
                },
            }
            actions.append(action)
            # Bulk-index in increments of BATCH_SIZE
            if len(actions) >= BATCH_SIZE:
                self._bulk_index(actions)
                actions = []

        if actions:
            self._bulk_index(actions)

    def _bulk_index(self, actions):
        try:
            success, errors = bulk(self.client, actions)
            print(f"[OpenSearchIndexer] Bulk indexed {success}, errors={errors}")
        except Exception as e:
            print("[OpenSearchIndexer] Bulk error:", e)

    def search(
        self, query_emb: np.ndarray, k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.client or query_emb.size == 0:
            print("[Error] Query embedding is empty!")
            return []

        if query_emb.shape[1] != EMBED_DIM:  # Ensure correct dimensionality
            print(
                f"[Error] Embedding shape mismatch! Expected ({EMBED_DIM},) but got {query_emb.shape}"
            )
            return []

        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (q_norm + 1e-9)
        vector = query_emb[0].tolist()
        # print(f"vector: {vector}")
        query_body = {
            "size": k,
            "query": {
                "knn": {
                    # "embedding": {"vector": vector, "k": k}
                    "field": "embedding",
                    "query_vector": vector,
                    "k": k,
                    "num_candidates": 100,
                }
            },
        }

        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            hits = resp["hits"]["hits"]
            return [(h["_source"], float(h["_score"])) for h in hits]
        except Exception as e:
            print("[OpenSearchIndexer] Search error:", e)
            return []


# ==============================================================================
# Basic Utils
# ==============================================================================
def basic_cleaning(text: str) -> str:
    return text.replace("\n", " ").strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    out = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        out.append(chunk.strip())
    return out


# ==============================================================================
# RAGModel
# ==============================================================================
class RAGModel:
    def __init__(self):
        self.os_indexer: Optional[OpenSearchIndexer] = None
        if os_client:
            self.os_indexer = OpenSearchIndexer(os_client, OPENSEARCH_INDEX_NAME)

    async def build_embeddings_from_scratch(self, pmc_dir: str):
        """
        Scans the local PMC text files, chunks them, and then uses
        batch-based embedding to index in OpenSearch.
        """
        if not self.os_indexer:
            print("[RAGModel] No OpenSearch client => cannot index.")
            return

        if self.os_indexer.has_any_data():
            print("[RAGModel] OpenSearch has data, skipping re-embedding.")
            return

        print("[RAGModel] Building embeddings from local text files...")
        files = os.listdir(pmc_dir)
        docs = []
        for fname in files:
            if fname.startswith("PMC") and fname.endswith(".txt"):
                path = os.path.join(pmc_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="latin-1") as f:
                        raw_text = f.read()

                cleaned = basic_cleaning(raw_text)
                chunks = chunk_text(cleaned, CHUNK_SIZE)
                for c in chunks:
                    docs.append({"doc_id": fname, "text": c})

        if not docs:
            print("[RAGModel] No local docs found.")
            return

        print(f"[RAGModel] Embedding {len(docs)} text chunks in batches...")
        # Extract just the text from each doc
        texts = [d["text"] for d in docs]

        # Embed all texts in asynchronous batches
        embed_array = await embed_texts_in_batches(texts, batch_size=BATCH_SIZE)

        # Index into OpenSearch
        self.os_indexer.add_embeddings(embed_array, docs)
        print("[RAGModel] Finished indexing in OpenSearch.")

    def os_search(self, query_emb: np.ndarray, top_k: int = 5):
        if not self.os_indexer:
            return []

        return self.os_indexer.search(query_emb, k=top_k)

    async def ask(self, query: str, top_k: int = 5) -> str:
        if not query or not query.strip():
            return "[ERROR] Empty query."

        # print(f"Query: {query}")
        # Embed query
        q_emb = embed_query_jina(query)
        # print(f"Query-Embeddings: {q_emb}")

        # Check Redis LFU cache
        cached = lfu_cache_get(q_emb)
        if cached:
            print("[RAGModel] Cache hit.")
            return cached

        # Determine intent => set how many docs
        intent_label = combined_intent_classification(query)
        effective_top_k = CATEGORY_TOPK_MAP.get(intent_label, top_k)

        # Special handling
        # if intent_label == "find_most_recent":
        #     recents = find_most_recent_papers(limit=effective_top_k)
        #     if not recents:
        #         return "No recent papers found."

        #     context_text = ""
        #     for r in recents:
        #         context_text += (
        #             f"--- Paper ID: {r['id']} ---\n"
        #             f"Title: {r['title']}\n"
        #             f"{r['abstract']}\n\n"
        #         )
        #     answer = await remote_summarize(context_text, query)
        #     lfu_cache_put(q_emb, answer)
        #     return answer

        # elif intent_label == "find_highly_cited":
        #     cits = find_highly_cited_papers(limit=effective_top_k)
        #     if not cits:
        #         return "No highly cited papers found."

        #     context_text = ""
        #     for c in cits:
        #         context_text += (
        #             f"--- Paper ID: {c['id']} ---\n"
        #             f"Title: {c['title']}\n"
        #             f"{c['abstract']}\n\n"
        #         )
        #     answer = await remote_summarize(context_text, query)
        #     lfu_cache_put(q_emb, answer)
        #     return answer

        # if intent_label == "count_documents":
        #     results = self.os_search(q_emb, effective_top_k)
        #     doc_ids = set(r[0]["doc_id"] for r in results)
        #     final = f"Found {len(doc_ids)} matching docs: {list(doc_ids)}"
        #     lfu_cache_put(q_emb, final)
        #     return final

        # Otherwise => KNN in OpenSearch
        results = self.os_search(q_emb, effective_top_k)
        if not results:
            return "No documents found for your query."

        doc_map = {}
        for doc_source, score in results:
            doc_id = doc_source["doc_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc_source["text"]
            else:
                doc_map[doc_id] += "\n" + doc_source["text"]

        print(f"[OpenSearchIndexer] Found {len(doc_map)} docs in search.")

        context_text = ""
        for doc_id, doc_content in doc_map.items():
            print(doc_id)
            context_text += f"--- Document: {doc_id} ---\n{doc_content}\n\n"

        # print(context_text)

        final_answer = await remote_answer(context_text, query)
        lfu_cache_put(q_emb, final_answer)
        return final_answer


async def remote_summarize(context_text: str, user_query: str, max_tokens=200) -> str:
    prompt_context = (
        "You are a helpful assistant. Summarize the following results.\n\n"
        f"Query: {user_query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Give a short summary or direct answer, referencing relevant Document IDs or Names if needed.\n"
    )
    return await call_remote_llm(
        context_text=prompt_context, user_query=user_query, max_tokens=max_tokens
    )


async def remote_answer(context_text: str, user_query: str, max_tokens=200) -> str:
    # prompt_context = (
    #     "### User Instructions:\n"
    #     "You are a helpful medical assistant. You have access to the following provided context "
    #     "which may contain references to documents (e.g., PMC51123.txt, etc.). "
    #     "Answer accurately using only this provided information or context. "
    #     "Mention document IDs for any references in your final answer. "
    #     "If there's insufficient info, just say so.\n\n"
    #     f"### Relevant Information or Context (Documents):\n{context_text}\n\n"
    # )

    prompt_context = (
        "Provide your answer strictly based on the provided context or information, following every instruction below extremely carefully:\n"
        "You must cite the Document ID or Name for any information you use. "
        "Your answer must follow this format: Direct Answer (at most 4 sentences). References (list any Document IDs from which the answer is extracted, e.g., Document XYZ). "
        "Your answer must be specific to the user query regardless of the provided context information. "
        "If insufficient information or if the provided context is not relevant to the user query, then clearly say so instead of giving a false response based on your own knowledge of these terms or the question. "
        "If the Document ID is 'PMC555957.txt', refer to it as 'Document PMC555957' (without the file extension). "
        "If multiple documents are relevant, cite each of them explicitly. "
        "Do NOT reveal your reasoning or chain of thought.\n"
        "--- End of Instructions ---\n\n"
        f"User Query: {user_query}\n"
        f"Context or Information:\n{context_text}\n"
    )
    return await call_remote_llm(
        context_text=prompt_context, user_query="", max_tokens=max_tokens
    )


# ==============================================================================
# FastAPI Setup
# ==============================================================================
app = FastAPI(
    title="Local RAG + Jina Embeddings + Local DB + Remote LLM", version="3.0"
)

rag_model: Optional[RAGModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_model
    rag_model = RAGModel()
    print(f"Model's Attention Config: {embedding_model.config.position_embedding_type}")
    await rag_model.build_embeddings_from_scratch(PMC_DIR)
    yield
    print("[Shutdown] Done.")


@app.post("/search")
async def search_opensearch_route(query: str = Body(...), top_k: int = 5):
    """Direct OS search endpoint with no LLM usage."""
    rm = rag_model
    if not rm:
        return {"error": "No RAG model."}

    q_emb = embed_query_jina(query)
    results = rm.os_search(q_emb, top_k)
    out = []
    for doc_source, score in results:
        out.append(
            {
                "doc_id": doc_source["doc_id"],
                "text": doc_source["text"][:200],
                "score": score,
            }
        )
    return {"query": query, "top_k": top_k, "results": out}


@app.post("/ask")
async def ask_route(query: str = Body(...), top_k: int = 5):
    """
    Full pipeline:
    1) Embed => 2) Classify => 3) Retrieve => 4) Possibly call remote LLM => Return answer
    """
    if not rag_model:
        return {"error": "RAG model not initialized."}

    answer = await rag_model.ask(query, top_k)
    return {"query": query, "answer": answer}


@app.get("/healthcheck")
def health():
    return {"status": "ok"}


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
