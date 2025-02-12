import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

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

# We maintain a maximum number of cached items.
# When exceeded, remove the least frequently used one.
REDIS_MAX_ITEMS = 1000
REDIS_CACHE_LIST = "query_cache_lfu"

# Embeddings
EMBED_DIM = 768
CHUNK_SIZE = 512
BATCH_SIZE = 128

# ------------------------------
# Device Selection for Embeddings
# ------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    inference_dtype = torch.float16  # Faster on most NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    inference_dtype = torch.float16  # MPS supports float16 as well
else:
    device = torch.device("cpu")
    inference_dtype = torch.float32  # CPU => stick to float32

# BioBERT model name (PubMed + MIMIC)
BIOBERT_MODEL_NAME = "dmis-lab/biobert-v1.1"

# Remote LLM endpoint
REMOTE_LLM_URL = os.getenv("REMOTE_LLM_URL", "")

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

# Directory for local text files
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
    """
    Retrieve from Redis if we find a sufficiently similar embedding.
    Increase freq if found.
    """
    cached_list = redis_client.lrange(REDIS_CACHE_LIST, 0, -1)
    if not cached_list:
        return None

    # Flatten to 1D for comparison
    query_vec = query_emb[0]
    best_sim = -1.0
    best_index = -1
    best_entry_data = None

    for i, item in enumerate(cached_list):
        entry = json.loads(item)
        emb_list = entry["embedding"]
        freq = entry.get("freq", 1)

        cached_emb = np.array(emb_list, dtype=np.float32)
        sim = cosine_similarity(query_vec, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_index = i
            best_entry_data = entry

    # Define a threshold
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
    """
    Insert new entry, remove LFU if over capacity.
    """
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


def find_most_recent_papers(limit: int = 5) -> List[Dict[str, Any]]:
    query = f"""
        SELECT id, title, abstract, publication_date
        FROM papers
        ORDER BY publication_date DESC
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
        print("[Postgres] find_most_recent_papers error:", e)
        return []


def find_highly_cited_papers(limit: int = 5) -> List[Dict[str, Any]]:
    query = f"""
        SELECT id, title, abstract, citation_count
        FROM papers
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
        print("[Postgres] find_highly_cited_papers error:", e)
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

    # pick best
    return max(doc.cats, key=doc.cats.get)


def fallback_classify(query: str) -> str:
    # Simple heuristics if spaCy fails
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


# Maps classification => how many docs to retrieve from OpenSearch
CATEGORY_TOPK_MAP = {
    "count_documents": 100,
    "retrieve_info": 5,
    "find_most_recent": 5,
    "find_highly_cited": 10,
    "other": 5,
}

# ==============================================================================
# BioBERT for Embeddings (Optimized, Balanced Accuracy)
# ==============================================================================
print("[Load] Initializing BioBERT for embeddings:", BIOBERT_MODEL_NAME)
biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)

# Load the model in mixed precision if GPU is available, else float32 on CPU
biobert_model = AutoModel.from_pretrained(
    BIOBERT_MODEL_NAME,
    torch_dtype=inference_dtype,
).to(device)
biobert_model.eval()


@torch.inference_mode()
def embed_text_bert(text: str) -> np.ndarray:
    """
    Generate a single 768-d vector from BioBERT via mean pooling.
    We run the forward pass in half precision on GPUs (float16) or float32 on CPU.
    Final embedding is stored in float32 to preserve similarity accuracy.
    """
    if not text.strip():
        return np.zeros((EMBED_DIM,), dtype=np.float32)

    inputs = biobert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    # Only enable autocast if we are on CUDA (NVIDIA) GPU (since its not supported on MPS at the moment):
    cast_enabled = device.type == "cuda"

    # Use autocast on CUDA, do nothing on MPS or CPU
    if cast_enabled:
        # Mixed precision on NVIDIA GPU
        with autocast("cuda", enabled=True):
            outputs = biobert_model(**inputs)
    else:
        # No autocast if MPS or CPU
        outputs = biobert_model(**inputs)

    # Mean pooling
    hidden = outputs.last_hidden_state  # shape: [1, seq_len, 768]
    emb = hidden.mean(dim=1).squeeze()  # shape: [768]

    # Move back to CPU and convert to float32
    emb_np = emb.detach().cpu().numpy().astype(np.float32)
    return emb_np


def embed_query_bert(query: str) -> np.ndarray:
    emb = embed_text_bert(query)
    return np.expand_dims(emb, axis=0)


# ==============================================================================
# Remote LLM Call
# ==============================================================================
async def call_remote_llm(
    context_text: str, user_query: str, max_tokens: int = 200
) -> str:
    """
    Calls the remote LLM endpoint (TGI or custom) with context + query,
    returns the final answer. We use httpx for async request.

    Expecting the remote service to parse:
      {
        "context": <string>,
        "query": <string>,
        "max_new_tokens": ...
      }

    and return JSON: {"answer": "..."}
    """
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
                            "parameters": {"m": 32, "ef_construction": 400},
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

        # Normalize embeddings for better KNN search
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
            return []
        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb = query_emb / (q_norm + 1e-9)
        vector = query_emb[0].tolist()

        query_body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
        }
        # query_body = {
        #     "knn": {
        #         "field": "embedding",  # The field where the vectors are stored
        #         "query_vector": vector,  # The normalized query vector
        #         "k": k,  # Number of nearest neighbors to return
        #         "num_candidates": 200,  # Controls search depth (improves accuracy)
        #     }
        # }

        try:
            resp = self.client.search(index=self.index_name, body=query_body)
            # resp = self.client.transport.perform_request(
            #     method="POST", url=f"/{self.index_name}/_knn_search", body=query_body
            # )
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
        if not self.os_indexer:
            print("[RAGModel] No OpenSearch client => cannot index.")
            return

        if self.os_indexer.has_any_data():
            print("[RAGModel] OpenSearch has data, skipping re-embedding.")
            return

        print("[RAGModel] Building embeddings from local text...")
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

        print(f"[RAGModel] Embedding {len(docs)} text chunks with BioBERT...")
        embed_list = []
        for d in docs:
            emb = embed_text_bert(d["text"])
            embed_list.append(emb)

        embed_array = np.array(embed_list, dtype=np.float32)
        self.os_indexer.add_embeddings(embed_array, docs)
        print("[RAGModel] Finished indexing in OpenSearch.")

    def os_search(self, query_emb: np.ndarray, top_k: int = 5):
        if not self.os_indexer:
            return []

        return self.os_indexer.search(query_emb, k=top_k)

    async def ask(self, query: str, top_k: int = 5) -> str:
        if not query.strip():
            return "[ERROR] Empty query."

        # 1) Embed locally with BioBERT
        q_emb = embed_query_bert(query)

        # 2) Redis LFU cache check
        cached = lfu_cache_get(q_emb)
        if cached:
            print("[RAGModel] Cache hit.")
            return cached

        # 3) Intent classification => skip LLM if not needed
        intent_label = combined_intent_classification(query)
        effective_top_k = CATEGORY_TOPK_MAP.get(intent_label, top_k)

        # 4) Handle special cases
        if intent_label == "find_most_recent":
            recents = find_most_recent_papers(limit=effective_top_k)
            if not recents:
                return "No recent papers found."

            context_text = ""
            for r in recents:
                context_text += (
                    f"--- Paper ID: {r['id']} ---\n"
                    f"Title: {r['title']}\n"
                    f"{r['abstract']}\n\n"
                )

            answer = await remote_summarize(context_text, query)
            lfu_cache_put(q_emb, answer)
            return answer

        elif intent_label == "find_highly_cited":
            cits = find_highly_cited_papers(limit=effective_top_k)
            if not cits:
                return "No highly cited papers found."

            context_text = ""
            for c in cits:
                context_text += (
                    f"--- Paper ID: {c['id']} ---\n"
                    f"Title: {c['title']}\n"
                    f"{c['abstract']}\n\n"
                )

            answer = await remote_summarize(context_text, query)
            lfu_cache_put(q_emb, answer)
            return answer

        elif intent_label == "count_documents":
            results = self.os_search(q_emb, effective_top_k)
            doc_ids = set(r[0]["doc_id"] for r in results)
            final = f"Found {len(doc_ids)} matching docs: {list(doc_ids)}"
            lfu_cache_put(q_emb, final)
            return final

        # 5) Otherwise => do KNN search in OpenSearch
        results = self.os_search(q_emb, effective_top_k)
        if not results:
            return "No documents found for your query."

        # Build context from docs
        doc_map = {}
        for doc_source, score in results:
            d_id = doc_source["doc_id"]
            if d_id not in doc_map:
                doc_map[d_id] = doc_source["text"]
            else:
                doc_map[d_id] += "\n" + doc_source["text"]

        context_text = ""
        for doc_id, content in doc_map.items():
            print(doc_id)
            context_text += (
                f"--- Document ID: {doc_id} ---\nDocument Content: {content}\n\n"
            )

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
        context_text=prompt_context, user_query="", max_tokens=max_tokens
    )


async def remote_answer(context_text: str, user_query: str, max_tokens=200) -> str:
    prompt_context = (
        "### Instructions:\n"
        "You are a helpful medical assistant. You have access to the following provided context "
        "which may contain references to documents (e.g., PMC51123.txt, PMC23124.txt, etc.). "
        "Your job is to answer the user's question accurately using only this provided information or context and nothing else. "
        "Mention the document IDs citing the specific term or discussing the subject (from which you extracted the user-query specific information) "
        "in your answer in the format: Answer Text -> Document ID/Name (without the file extension like '.txt'). "
        "Do not include the prompt or your chain-of-thought in your response! Only provide the final answer using the provided context and nothing more. "
        "If there is not enough info, just say so!\n\n"
        f"### Relevant Information or Context (Documents):\n{context_text}\n\n"
        # "### Response:\n"
        "Answer accurately only using this information that was retrieved from the database!"
    )
    return await call_remote_llm(
        context_text=prompt_context, user_query=user_query, max_tokens=max_tokens
    )


# ==============================================================================
# FastAPI Setup
# ==============================================================================
app = FastAPI(title="Local RAG + BioBERT + Local DB + Remote LLM)", version="6.0")

rag_model: Optional["RAGModel"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_model
    rag_model = RAGModel()
    # Possibly embed local docs if needed
    await rag_model.build_embeddings_from_scratch(PMC_DIR)
    yield
    print("[Shutdown] Done.")


@app.post("/search")
async def search_opensearch_route(query: str = Body(...), top_k: int = 5):
    """Direct OS search endpoint with no LLM usage."""
    rm = rag_model
    if not rm:
        return {"error": "No RAG model."}

    q_emb = embed_query_bert(query)
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
