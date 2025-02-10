import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from fastapi import FastAPI, Body
import uvicorn
import redis

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

import asyncio
from contextlib import asynccontextmanager

import httpx  # for Ollama's embedding calls
import openai  # for GPT-4o generation

import spacy  # Load the spaCy text categorizer
from spacy.tokens import Doc

# ==============================================================================
# Configuration Constants
# ==============================================================================

# Ollama specifics for embeddings
OLLAMA_API_URL = "http://localhost:11434/api"
EMBED_MODEL_NAME = "jina/jina-embeddings-v2-base-de"  # local Ollama embedding model

# OpenAI specifics for generation
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)
openai.api_key = OPENAI_API_KEY
OPENAI_CHAT_MODEL = "gpt-4o"

BATCH_SIZE = 128
CHUNK_SIZE = 512
EMBED_DIM = 768
MAX_EMBED_CONCURRENCY = 5  # Concurrency limit to avoid overwhelming Ollama

REDIS_HOST = "localhost"
REDIS_PORT = 6379
MAX_QUERY_CACHE = 1000
CACHE_SIM_THRESHOLD = 0.96  # Cosine similarity threshold for cache lookup

PMC_DIR = "./PMC"

# We load the text-cat model from a local directory (adjust path if needed).
SPACY_MODEL_PATH = "./model_nlu"

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
    """Obtain an embedding for a single query from Ollama (local)."""
    if not query.strip():
        return np.array([])

    emb = await ollama_embed_text(query, EMBED_MODEL_NAME)
    return np.array([emb], dtype=np.float32)


# ==============================================================================
# OpenAI for Remote Generation
# ==============================================================================
async def openai_generate_text(
    prompt: str,
    model: str = OPENAI_CHAT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 256,
) -> str:
    """
    Request text generation from the new openai.chat.completions interface.
    Returns the generated text content.
    """
    loop = asyncio.get_running_loop()

    def _run_openai():
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Follow the user's instructions exactly.\n"
                        "If you reference a document, use the phrase 'Document <ID>' without file extensions.\n"
                        "Do not reveal your chain of thought."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Extract and return the content of the first completion
        return response.choices[0].message.content.strip()

    # Offload the synchronous call to a thread pool to avoid blocking
    return await loop.run_in_executor(None, _run_openai)


async def call_llm(prompt: str) -> str:
    """
    A wrapper function for the answer generation step using OpenAI.
    """
    return await openai_generate_text(prompt)


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
                            "parameters": {
                                "m": 32,
                                "ef_construction": 400,
                                "ef_search": 200,
                            },
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
# SpaCy Classification
# ==============================================================================
spacy_textcat_nlp = None


def classify_query(query: str) -> str:
    """
    Classifies the user query using the loaded spaCy text-cat model.
    Returns the best label, or 'retrieve_info' fallback if the model isn't loaded.
    """
    if not spacy_textcat_nlp:
        return "retrieve_info"  # default if model missing
    doc = spacy_textcat_nlp(query)
    cats = doc.cats
    best_label = max(cats, key=cats.get)
    return best_label


# A mapping from category -> top_k
CATEGORY_TOPK_MAP = {
    "count_documents": 100,
    "retrieve_info": 5,
    "summarize_documents": 5,
    "compare_topics": 5,
    "list_documents": 20,
    "extract_specific_info": 5,
    "answer_directly": 3,
    "find_most_recent": 5,
    "find_highly_cited": 10,
    "document_metadata": 10,
    "document_references": 10,
    "filter_by_condition": 5,
    "find_similar_documents": 5,
}


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
        await loop.run_in_executor(None, self.os_indexer.add_embeddings, embs, all_docs)
        print("[RAGModel] Finished embedding & indexing data in OpenSearch.")

    def os_search(self, query_emb: np.ndarray, top_k=5):
        if not self.os_indexer:
            return []

        return self.os_indexer.search(query_emb, k=top_k)

    async def ask(self, query: str, top_k: int = 5) -> str:
        """
        Process a user query:
          1. Get its embedding via Ollama
          2. Check Redis cache and if found return the response
          3. Classify the query with the spaCy model
          4. Adjust top_k or other logic based on classification
          5. If not cached in Redis, search OpenSearch
          6. Call LLM for the final answer
          7. Cache and return the answer
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

        # 3) Classify the query with spaCy
        spacy_cat_label = classify_query(query)

        # If category is count, then do not load the documents them

        # 4) Adjust the top_k parameter based on the predicted category
        effective_top_k = CATEGORY_TOPK_MAP.get(spacy_cat_label, top_k)
        loop = asyncio.get_running_loop()

        if spacy_cat_label == "count_documents":
            # The user wants to know how many documents talk about a subject
            # We'll fetch documents in BATCHES and for each batch:
            # Summarize them in a prompt
            # Ask the LLM to count how many are truly relevant
            # Then aggregate these partial counts

            BATCH_SIZE_COUNT = 30
            doc_offset = 0
            total_relevant_count = 0
            all_relevant_docs = []

            while doc_offset < effective_top_k:
                # We'll retrieve the next batch of documents by specifying the offset in search
                batch_results = await loop.run_in_executor(
                    None,
                    self._search_with_offset,
                    query_emb,
                    BATCH_SIZE_COUNT,
                    doc_offset,
                )
                if not batch_results:
                    # No more results
                    break

                # Build a chunk of context
                doc_map = {}
                for doc_dict, score in batch_results:
                    doc_id = doc_dict["doc_id"]
                    text_chunk = doc_dict["text"]
                    if doc_id not in doc_map:
                        doc_map[doc_id] = text_chunk
                    else:
                        doc_map[doc_id] += "\n" + text_chunk

                # Form a smaller prompt instructing the LLM:
                #   "Out of these docs, which truly discuss <query subject>? Return count + doc IDs"
                batch_context = ""
                for doc_id, doc_content in doc_map.items():
                    batch_context += f"--- Document: {doc_id} ---\n{doc_content}\n\n"

                count_prompt = (
                    "You are tasked with counting how many of the following documents truly discuss the user query.\n"
                    "Return the total count of relevant documents, and list out the Document IDs that are relevant.\n"
                    "Ignore or discard any irrelevant or partially relevant docs.\n\n"
                    f"User Query for Counting:\n{query}\n\n"
                    f"Documents:\n{batch_context}\n\n"
                    "You must respond in the following JSON format EXACTLY:\n"
                    "{\n"
                    '  "relevant_count": <integer>,\n'
                    '  "doc_ids": ["Doc1","Doc2",...]\n'
                    "}\n"
                    "Do not add anything else. Just the JSON in that format."
                )

                partial_answer = await call_llm(count_prompt)
                # Now parse partial_answer JSON
                try:
                    partial_data = json.loads(partial_answer)
                    batch_count = partial_data.get("relevant_count", 0)
                    batch_docs = partial_data.get("doc_ids", [])
                    total_relevant_count += batch_count
                    all_relevant_docs.extend(
                        batch_docs if isinstance(batch_docs, list) else []
                    )
                except json.JSONDecodeError:
                    # If LLM response is invalid, skip or handle error
                    pass

                # Move offset
                doc_offset += BATCH_SIZE_COUNT

            # Now we have total_relevant_count + all_relevant_docs
            # Build a final answer
            final_count_response = (
                f"Found {total_relevant_count} relevant documents that truly discuss your topic.\n"
                f"Relevant Document IDs: {all_relevant_docs}"
            )
            # Cache in Redis
            store_query_in_redis(query_emb, final_count_response)
            return final_count_response

        # 5) Search OpenSearch for other categories as usual
        results = await loop.run_in_executor(
            None, self.os_search, query_emb, effective_top_k
        )

        # 6) Prepare context & call LLM
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
            context_text += f"--- Document: {doc_id} ---\n{doc_content}\n\n"

        final_prompt = (
            "INSTRUCTIONS: You must cite the **Document ID or Name** for any information you use. "
            "Your answer must follow this format: Direct Answer (at most 4 sentences)\nReferences (list any Document IDs from which the answer is extracted, e.g., Document XYZ). "
            "Your answer must be specific to the user query. "
            "If the Document ID is 'PMC555957.txt', refer to it as 'Document PMC555957' (no file extension). "
            "If multiple documents are relevant, cite them explicitly. "
            "Do NOT reveal your chain of thought.\n\n"
            f"User Query:\n{query}\n\n"
            f"Context:\n{context_text}\n"
            "--- End of context ---\n\n"
            "Provide your answer strictly based on the above information. If insufficient, say so."
        )

        answer = await call_llm(final_prompt)

        # 7) Cache in Redis
        store_query_in_redis(query_emb, answer)
        return answer

    def _search_with_offset(self, query_emb: np.ndarray, batch_size: int, offset: int):
        """
        This internal helper function uses 'from' and 'size' to retrieve
        documents from OpenSearch in a paginated manner, so we can read them in small batches.
        """
        if not self.os_indexer or query_emb.size == 0:
            return []

        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        normalized_emb = query_emb / (q_norm + 1e-9)
        vector = normalized_emb[0].tolist()
        # We use the 'knn' + 'from' and 'size' approach.
        # Alternatively, could do a normal 'script_score' approach or find a workaround.
        # If pure KNN doesn't allow offset, can do a different approach or ignore 'offset' for now.

        query_body = {
            "size": batch_size,
            "from": offset,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": batch_size + offset,  # an approximation
                    }
                }
            },
        }
        try:
            resp = self.os_indexer.client.search(
                index=self.os_indexer.index_name, body=query_body
            )
            hits = resp["hits"]["hits"]
            results = []
            for h in hits:
                doc_score = h["_score"]
                doc_source = h["_source"]
                results.append((doc_source, float(doc_score)))
            return results
        except Exception as e:
            print(f"[search_with_offset] Error: {e}")
            return []


# ==============================================================================
# FastAPI Application Setup
# ==============================================================================
app = FastAPI(
    title="RAG with LLMs + OpenSearch + Redis + spaCy Classification",
    version="3.2",
    description="RAG pipeline to answer queries using Ollama's embedding, OpenAI LLM, OpenSearch ANN, Redis cache, and spaCy text categorization.",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Lifespan] Initializing spaCy text-cat model + RAGModel...")
    global spacy_textcat_nlp
    global rag_model

    # Load spaCy text-cat model
    try:
        spacy_textcat_nlp = spacy.load(SPACY_MODEL_PATH)
        print("[Lifespan] spaCy text-cat model loaded successfully.")
    except Exception as e:
        print(f"[Lifespan] Failed to load spaCy text-cat model: {e}")
        spacy_textcat_nlp = None

    rag_model = RAGModel()
    await rag_model.build_embeddings_from_scratch(PMC_DIR)
    print("[Lifespan] RAGModel is ready.")

    yield  # run the application

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

    # We can also run text classification here if we want to set top_k
    predicted_cat = classify_query(query)
    effective_top_k = CATEGORY_TOPK_MAP.get(predicted_cat, top_k)

    # Embed query text
    q_emb = await embed_query(query)
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, rm.os_search, q_emb, effective_top_k)
    return {
        "query": query,
        "predicted_category": predicted_cat,
        "top_k": effective_top_k,
        "results": [
            {"doc_id": r[0]["doc_id"], "text": r[0]["text"], "score": r[1]}
            for r in results
        ],
    }


@app.post("/ask")
async def ask_route(query: str = Body(...), top_k: int = 5):
    """
    API endpoint that processes a user query through the entire RAG pipeline:
    classification + embedding + knn-search + LLM generation.
    """
    rm = get_rag_model()
    if not rm:
        return {"error": "RAG Model not initialized"}

    answer = await rm.ask(query, top_k)
    return {"query": query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
