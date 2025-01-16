import os
import numpy as np
from typing import List, Tuple, Optional

# FastAPI / Uvicorn
from fastapi import FastAPI, Body, BackgroundTasks, Depends
import uvicorn

# OpenAI Client Setup
from openai import OpenAI

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "",
)
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o"
EMBED_DIM = 1536


def embed_texts(texts: List[str]) -> np.ndarray:
    # Generate embeddings for a list of texts using OpenAI APIs.
    if not texts:
        return np.array([])

    all_embeddings = []
    BATCH_SIZE = 1000

    try:
        # Batch-process the chunks of text for embedding generation
        client = OpenAI(api_key=OPENAI_API_KEY)
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            batch_embs = [r.embedding for r in resp.data]
            all_embeddings.extend(batch_embs)
    except Exception as e:
        print(f"[ERROR] Embedding failure: {e}")
        return np.array([])

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embedding for a query string using the same OpenAI embedding model that was used
    to create document embeddings.
    Example response: {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [
                    -0.006929283495992422,
                    -0.005336422007530928,
                    -4.547132266452536e-05,
                    -0.024047505110502243
                ],
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }
    """
    if not query:
        return np.array([])

    try:
        # Generate Query embeddings
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
        emb = resp.data[0].embedding
        return np.array([emb], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Query embedding failure: {e}")
        return np.array([])


def call_llm_gpt(prompt: str) -> str:
    # Call GPT model (gpt-4o or other)
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            store=False,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM ERROR] {e}"


# FAISS (using in-memory store for storing indices on local machine)
import faiss


class FaissIndexer:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.docstore = []

    def add_embeddings(self, embeddings: np.ndarray, docs: List[str]):
        # Add embeddings to FAISS index
        if embeddings.size == 0:
            print("[FaissIndexer] No embeddings to index.")
            return

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"[FaissIndexer] Dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)
        self.docstore.extend(docs)

    def search(self, query_emb: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        # Search for top k docs via FAISS L2 distance using the embeddings
        if self.index.ntotal == 0 or query_emb.size == 0:
            return []

        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.docstore):
                text = self.docstore[idx]
                results.append((text, float(dist)))

        print(f"[FAISSIndexer] Found {len(results)} docs in search.")
        return results

    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def get_all_vectors_and_texts(self) -> Tuple[np.ndarray, List[str]]:
        """
        Returns all (embeddings, doc_texts) stored in FAISS in memory.
        - FAISS stores vectors in an internal structure. We use index.reconstruct(i)
          to get the i-th vector.
        - We retrieve all vectors from 0 ... (ntotal-1).
        - docstore is in sync, so docstore[i] corresponds to the i-th vector.

        it extracts them from the FAISS in-memory store.
        """
        total = self.index.ntotal
        if total == 0:
            return np.array([]), []

        # Reconstruct each embedding
        embs = []
        for i in range(total):
            vec = self.index.reconstruct(i)  # get the i-th vector/embedding
            embs.append(vec)

        embs_np = np.array(embs, dtype=np.float32)
        return embs_np, self.docstore[:]


# Elastic Search Setup
from elasticsearch import Elasticsearch, helpers

ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY", "")
ELASTIC_INDEX_NAME = "medical-search-index"

es_client: Optional[Elasticsearch] = None
try:
    es_client = Elasticsearch(
        "https://",
        api_key=ELASTIC_API_KEY,
    )
    if not es_client.ping():
        raise ValueError("Elasticsearch ping failed - invalid credentials or host.")

    if not es_client.indices.exists(index=ELASTIC_INDEX_NAME):
        resp = es_client.indices.create(
            index=ELASTIC_INDEX_NAME,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIM,
                        "similarity": "cosine",
                    },
                }
            },
        )
        print(
            f"[INFO] Created '{ELASTIC_INDEX_NAME}' index with cosine similarity: {resp}"
        )
    else:
        print(f"[INFO] Elasticsearch index '{ELASTIC_INDEX_NAME}' is ready.")
except Exception as e:
    print(f"[WARNING] Elasticsearch not initialized: {e}")
    es_client = None


class ElasticsearchIndexer:
    def __init__(self, es: Elasticsearch, index_name: str):
        self.es = es
        self.index_name = index_name

    def has_any_data(self) -> bool:
        # Returns True if index contains any documents else False.
        if not self.es:
            return False
        try:
            resp = self.es.count(index=self.index_name)
            return resp["count"] > 0
        except:
            return False

    def add_embeddings(self, embeddings: np.ndarray, docs: List[str]):
        # Adds doc embeddings into Elasticsearch using bulk() API
        if not self.es or embeddings.size == 0:
            print("[ElasticsearchIndexer] No embeddings or Elasticsearch client.")
            return

        BATCH_SIZE = 100
        actions = []
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": f"doc_{i}",
                    "_source": {"text": doc, "embedding": emb.tolist()},
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
            # Attempt the bulk indexing operation
            helpers.bulk(self.es, actions)
            print(f"[ElasticsearchIndexer] Inserted {len(actions)} docs.")
        except helpers.BulkIndexError as bulk_error:
            print(f"[ElasticsearchIndexer] Bulk insert error: Some documents failed.")
        except Exception as e:
            print(f"[ElasticsearchIndexer] Unexpected error during bulk index. {e}")

    def search(self, query_emb: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search in Elasticsearch using the given query embeddings for the relevant document chunks.
        Official Documentation for Search:
        client.search(
            index="my-index",
            size=3,
            query={
              "knn": {
                "field": "embedding",
                "query_vector": [...],
                "k": 10
              }
            }
        )
        """

        if not self.es or query_emb.size == 0:
            return []

        vector = query_emb[0].tolist()
        try:
            resp = self.es.search(
                index=self.index_name,
                size=k,
                query={"knn": {"field": "embedding", "query_vector": vector, "k": k}},
            )
            hits = resp["hits"]["hits"]
            results = []
            for h in hits:
                doc_text = h["_source"]["text"]
                doc_score = h["_score"]
                results.append((doc_text, float(doc_score)))

            print(f"[ElasticsearchIndexer] Found {len(results)} docs in search.")
            return results
        except Exception as e:
            print(f"[ElasticsearchIndexer] Search error: {e}")
            return []

    def fetch_all_docs(self) -> Tuple[np.ndarray, List[str]]:
        # Fetch all docs from this ES index. Returns (embeddings in nparray format, texts in a list like [text, text, ...])

        if not self.es:
            return np.array([]), []

        # Scroll or search approach - a simple match all with scroll
        embeddings = []
        texts = []
        try:
            resp = self.es.search(
                index=self.index_name, scroll="2m", size=1000, query={"match_all": {}}
            )
            sid = resp["_scroll_id"]
            hits = resp["hits"]["hits"]

            while hits:
                for h in hits:
                    source = h["_source"]
                    texts.append(source["text"])
                    emb = source["embedding"]
                    embeddings.append(emb)

                resp = self.es.scroll(scroll_id=sid, scroll="2m")
                sid = resp["_scroll_id"]
                hits = resp["hits"]["hits"]
        except Exception as e:
            print(f"[ElasticsearchIndexer] fetch_all_docs error: {e}")
            return np.array([]), []

        return np.array(embeddings, dtype=np.float32), texts


# File Paths
PMC_DIR = "./PMC"


# Basic Pre-processing
def basic_cleaning(text: str) -> str:
    return text.replace("\n", " ").strip()


def chunk_text(text: str, chunk_size=150) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk.strip())
    return chunks


# RAG model and background replication


def replicate_faiss_to_es(
    faiss_indexer: FaissIndexer, es_indexer: ElasticsearchIndexer
):
    # Background task to replicate data from FAISS => ES
    print("[BackgroundTasks] Finished replicating FAISS => ES.")
    emb, docs = faiss_indexer.get_all_vectors_and_texts()
    es_indexer.add_embeddings(emb, docs)


def replicate_es_to_faiss(
    es_indexer: ElasticsearchIndexer, faiss_indexer: FaissIndexer
):
    # Background task to replicate data from ES => FAISS
    print("[BackgroundTasks] Finished replicating ES => FAISS.")
    es_emb, es_docs = es_indexer.fetch_all_docs()
    faiss_indexer.add_embeddings(es_emb, es_docs)


class RAGModel:
    """
    1. Checks existing embeddings in FAISS + ES + Pinecone (not in the current version of the code)
    2. Generates new embeddings only if both are empty
    3. If one is non-empty and the other is empty, replicate embeddings
    4. Searches individually or combined
    5. Answer-generation with GPT-4o
    """

    def __init__(
        self,
        pmc_dir: str,
        use_elastic: bool = True,
        background_tasks: Optional[BackgroundTasks] = None,
    ):
        self.faiss_indexer = FaissIndexer(dim=EMBED_DIM)
        self.elasticsearch_indexer: Optional[ElasticsearchIndexer] = None
        # replication_src: 0 => FAISS->ES, 1 => ES->FAISS, -1 => no replication
        self.replication_src: int = -1

        if use_elastic and es_client:
            self.elasticsearch_indexer = ElasticsearchIndexer(
                es_client, ELASTIC_INDEX_NAME
            )

        # Decide if we need new embeddings at all
        faiss_empty = self.faiss_indexer.is_empty()
        es_empty = True
        if self.elasticsearch_indexer and self.elasticsearch_indexer.has_any_data():
            es_empty = False

        # If both FAISS + ES have data, do nothing
        if not faiss_empty and not es_empty:
            print("[RAGModel] FAISS and ES both have data. No re-embedding.")
            return

        # If FAISS is not empty but ES is empty => pull from FAISS and store in ES
        if not faiss_empty and es_empty:
            print(
                "[RAGModel] FAISS has data, ES is empty => pushing FAISS embeddings to ES."
            )
            self.replication_src = 0  # FAISS => ES
            if self.elasticsearch_indexer and background_tasks:
                background_tasks.add_task(
                    replicate_faiss_to_es,
                    self.faiss_indexer,
                    self.elasticsearch_indexer,
                )
            elif self.elasticsearch_indexer:
                replicate_faiss_to_es(self.faiss_indexer, self.elasticsearch_indexer)

            return

        # If FAISS is empty but ES is not => pull from ES and build FAISS
        if faiss_empty and not es_empty:
            print(
                "[RAGModel] ES has data, FAISS is empty => pulling ES embeddings to FAISS."
            )
            self.replication_src = 1  # ES => FAISS
            if self.elasticsearch_indexer and background_tasks:
                background_tasks.add_task(
                    replicate_es_to_faiss,
                    self.elasticsearch_indexer,
                    self.faiss_indexer,
                )
            elif self.elasticsearch_indexer:
                replicate_es_to_faiss(self.elasticsearch_indexer, self.faiss_indexer)

            return

        # Otherwise => both are empty => embed from scratch
        print(
            "[RAGModel] Both FAISS & ES are empty => generating embeddings from scratch."
        )
        self._build_embeddings_from_scratch(pmc_dir)

    def _build_embeddings_from_scratch(self, pmc_dir: str):
        # load data
        data_texts = self._load_pmc_data(pmc_dir)
        data_texts = [txt for txt in data_texts if txt.strip()]

        # chunk + clean
        cleaned = [basic_cleaning(t) for t in data_texts]
        chunks = []
        for c in cleaned:
            chunks.extend(chunk_text(c))

        if not chunks:
            print("[RAGModel] No text found.")
            return

        print(f"[RAGModel] Generating embeddings for {len(chunks)} chunks.")
        embs = embed_texts(chunks)

        print("[RAGModel] Adding data to FAISS index.")
        self.faiss_indexer.add_embeddings(embs, chunks)

        if self.elasticsearch_indexer:
            print("[RAGModel] Adding data to Elasticsearch index.")
            self.elasticsearch_indexer.add_embeddings(embs, chunks)

    def _load_pmc_data(self, directory: str) -> List[str]:
        results = []
        if not os.path.isdir(directory):
            print(f"[WARNING] PMC dir '{directory}' not found.")
            return results

        # i = 1
        for fname in os.listdir(directory):
            # if i > 100:
            #     break

            # i += 1
            if fname.startswith("PMC") and fname.endswith(".txt"):
                path = os.path.join(directory, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        results.append(f.read())
                except UnicodeDecodeError:
                    try:
                        with open(path, "r", encoding="latin-1") as f:
                            results.append(f.read())
                        print(f"[WARNING] '{fname}' read with 'latin-1' fallback.")
                    except:
                        print(f"[ERROR] Could not decode '{fname}'. Skipping.")

        return results

    # Searching
    def faiss_search_with_emb(self, query_emb: np.ndarray, top_k=3):
        return self.faiss_indexer.search(query_emb, k=top_k)

    def es_search_with_emb(self, query_emb: np.ndarray, top_k=3):
        if not self.elasticsearch_indexer:
            return []

        return self.elasticsearch_indexer.search(query_emb, k=top_k)

    def combined_search(self, query_emb: np.ndarray, top_k=3):
        """
        If replication is in progress from FAISS => ES, that means ES is partial => skip ES results.
        If replication is in progress from ES => FAISS, that means FAISS is partial => skip FAISS results.
        Otherwise search both and merge the results.

        FAISS => (text, distance) => convert distance => "score"
        ES => (text, score) and Sort by descending score
        """

        results = []
        # ES
        if self.replication_src == -1 or self.replication_src == 1:
            # if we are not replicating, or replicating from ES => FAISS,
            # then ES has the complete data, we do an ES search
            print(
                "[RAGModel] Replication ES=>FAISS in progress. Hence, skipping FAISS in combined search."
            )
            es_res = self.es_search_with_emb(query_emb, top_k)
            results.extend(es_res)

        # FAISS
        if self.replication_src == -1 or self.replication_src == 0:
            # if not replicating, or replicating from FAISS => ES
            # then FAISS has the data
            print(
                "[RAGModel] Replication FAISS=>ES in progress. Hence, skipping Elasticsearch in combined search."
            )
            faiss_res = self.faiss_search_with_emb(query_emb, top_k)
            # Convert distance => score
            faiss_converted = [(txt, 1.0 - dist) for (txt, dist) in faiss_res]
            results.extend(faiss_converted)

        # Sort descending by "score"
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def ask(self, query_emb: np.ndarray, original_query: str, top_k=3) -> str:
        """
        Use a pre-computed embedding for the query to retrieve chunks, then LLM.
        Augmentation and LLM query-answering
        1) Retrieve top_k doc chunks
        2) Combine them with the user query
        3) Call GPT-4o to get final answer
        """

        res = self.combined_search(query_emb, top_k=top_k)
        context_docs = [r[0] for r in res]

        context_text = "\n\n".join(context_docs)
        final_prompt = (
            f"User query:\n{original_query}\n\n"
            f"Relevant context:\n{context_text}\n\n"
            "Given the context, answer as helpfully as possible using only that. Directly jump to the answer without mentioning about the context given. But if insufficient context, say so."
        )
        return call_llm_gpt(final_prompt)


from contextlib import asynccontextmanager


# FastAPI integrated for REST APIs
app = FastAPI(
    title="RAG with OpenAI LLMs and embedding model",
    version="1.0.0",
    description="End-to-end RAG pipeline using FAISS, Elasticsearch, and OpenAI LLM GPT-4o + Embedding Model like text-embedding-ada-002.",
    lifespan=lambda app: lifespan(app),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifespan event handler to initialize and clean up resources if needed.
    print("[Lifespan] Server is starting up...")
    background_tasks = BackgroundTasks()

    # Instantiate RAGModel and store it in app.state
    app.state.rag_model = RAGModel(
        pmc_dir="./PMC", use_elastic=True, background_tasks=background_tasks
    )
    print("[Lifespan] RAGModel instantiated and replication tasks started if any.")
    yield
    # When the app is shutting down:
    print("[Lifespan] Server is shutting down...")


# Dependency to retrieve the RAGModel instance from app.state
def get_rag_model() -> RAGModel:
    if not hasattr(app.state, "rag_model"):
        raise RuntimeError("RAGModel not initialized.")
    return app.state.rag_model


@app.post("/search/faiss")
def search_faiss_route(
    query: str = Body(...), top_k: int = 3, rag_model: RAGModel = Depends(get_rag_model)
):
    # FAISS search with the embeddings
    q_emb = embed_query(query)
    results = rag_model.faiss_search_with_emb(q_emb, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": [{"text": r[0], "distance": r[1]} for r in results],
    }


@app.post("/search/elasticsearch")
def search_es_route(
    query: str = Body(...), top_k: int = 3, rag_model: RAGModel = Depends(get_rag_model)
):
    # Elastic-search with the embeddings
    q_emb = embed_query(query)
    results = rag_model.es_search_with_emb(q_emb, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": [{"text": r[0], "score": r[1]} for r in results],
    }


@app.post("/search/combined")
def search_combined_route(
    query: str = Body(...), top_k: int = 3, rag_model: RAGModel = Depends(get_rag_model)
):
    # Embed query once and combined search with the embeddings
    q_emb = embed_query(query)
    results = rag_model.combined_search(q_emb, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": [{"text": r[0], "score": r[1]} for r in results],
    }


@app.post("/ask")
def query_text(
    query: str = Body(...), top_k: int = 3, rag_model: RAGModel = Depends(get_rag_model)
):
    # Query -> Query Embedding -> Retrieval from ElasticSearch + FAISS -> Augmented -> GPT LLM.
    q_emb = embed_query(query)
    answer = rag_model.ask(q_emb, query, top_k=top_k)
    return {"query": query, "answer": answer}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
