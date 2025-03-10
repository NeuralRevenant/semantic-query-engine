import os
from dotenv import load_dotenv
import shutil
import asyncio
import uvicorn
import numpy as np
import time
import pathlib
from typing import List

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv()

BASE_UPLOAD_DIR = "uploads"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

# Ollama embedding configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
EMBED_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
MAX_EMBED_CONCURRENCY = 5  # concurrency for embedding calls
BATCH_SIZE = 64  # process 64 chunks per batch
CHUNK_SIZE = 512  # split text into ~512-word chunks
EMBED_DIM = 1024  # embedding dimension

# Base index name for user-specific indexes
BASE_OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")

# OpenSearch connection
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))

###############################################################################
# FastAPI Application
###############################################################################
app = FastAPI(
    title="User-Specific Plain Text Upload & Embedding",
    version="1.0.0",
    description=(
        "A microservice that:\n"
        "1. Accepts a text file + user_id.\n"
        "2. Derives a single doc_id from the file name + timestamp.\n"
        "3. Stores the file in 'uploads/{user_id}/doc_id.ext'.\n"
        "4. Chunks & embeds the text.\n"
        "5. Indexes all chunks under the same doc_id in the user-specific OpenSearch index.\n"
    ),
)

###############################################################################
# OpenSearch Setup
###############################################################################
try:
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
    )
except Exception as e:
    print(f"[ERROR] Could not connect to OpenSearch: {e}")
    os_client = None


def init_user_index(user_id: str):
    """
    Create user-specific index <BASE_OPENSEARCH_INDEX_NAME>-<user_id> if it doesn't exist.
    """
    if not os_client:
        print("[WARNING] No OpenSearch client => skipping index creation.")
        return

    index_name = f"{BASE_OPENSEARCH_INDEX_NAME}-{user_id}"
    if os_client.indices.exists(index_name):
        print(f"[INFO] Index '{index_name}' already exists.")
        return

    try:
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    # The single doc_id that identifies the file
                    "doc_id": {"type": "keyword"},
                    # The chunk text
                    "text": {"type": "text"},
                    # The embedding for that chunk
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBED_DIM,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "parameters": {"m": 64, "ef_construction": 500},
                        },
                    },
                }
            },
        }
        os_client.indices.create(index=index_name, body=index_body)
        print(f"[INFO] Created user-specific index '{index_name}'.")
    except Exception as e:
        print(f"[ERROR] Failed creating index '{index_name}': {e}")


###############################################################################
# Chunking
###############################################################################
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits large text into ~chunk_size-word chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_str = " ".join(words[i : i + chunk_size])
        chunks.append(chunk_str.strip())
    return chunks


###############################################################################
# Embedding Helpers
###############################################################################
async def ollama_embed_text(text: str, model: str = EMBED_MODEL_NAME) -> List[float]:
    """
    Request an embedding from Ollama via HTTP POST.
    """
    if not text.strip():
        return [0.0] * EMBED_DIM

    async with httpx.AsyncClient() as client:
        payload = {"model": model, "prompt": text, "stream": False}
        try:
            resp = await client.post(
                f"{OLLAMA_API_URL}/embeddings", json=payload, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding", [])
            if len(emb) != EMBED_DIM:
                print(
                    f"[WARNING] Mismatch embedding size. Expected {EMBED_DIM}, got {len(emb)}"
                )
            return emb
        except Exception as exc:
            print(f"[ERROR] Ollama embedding error: {exc}")
            return [0.0] * EMBED_DIM


async def embed_texts_in_batches(texts: List[str]) -> np.ndarray:
    """
    Concurrently embed text chunks in small batches. Returns [len(texts), EMBED_DIM] array.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    all_embeddings = []
    sem = asyncio.Semaphore(MAX_EMBED_CONCURRENCY)

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        async def embed_single(txt: str) -> List[float]:
            async with sem:
                return await ollama_embed_text(txt)

        tasks = [embed_single(t) for t in batch]
        results = await asyncio.gather(*tasks)
        all_embeddings.extend(results)

    return np.array(all_embeddings, dtype=np.float32)


###############################################################################
# Bulk Indexing (All Chunks => Single doc_id)
###############################################################################
def bulk_index_embeddings(
    user_id: str, doc_id: str, embeddings: np.ndarray, chunks: List[str]
):
    """
    Bulk index chunk embeddings for a single doc_id into user-specific index:
      - Each chunk is a separate doc in the index
      - `_id` = f"{doc_id}_{chunk_index}"
      - `_source.doc_id` = doc_id
      - `_source.text` = chunk
      - `_source.embedding` = the chunk’s embedding
    """
    if not os_client or embeddings.size == 0:
        print("[ERROR] Missing OpenSearch client or embeddings => cannot index.")
        return

    index_name = f"{BASE_OPENSEARCH_INDEX_NAME}-{user_id}"
    init_user_index(user_id)  # ensure user index is created

    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-9)

    actions = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings_normed)):
        # keep _id unique by appending the chunk idx
        open_search_id = f"{doc_id}_{i}"
        source_doc = {
            "doc_id": doc_id,
            "text": chunk,
            "embedding": emb.tolist(),
        }

        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": open_search_id,
                "_source": source_doc,
            }
        )

        if len(actions) >= BATCH_SIZE:
            try:
                success, errors = bulk(os_client, actions)
                print(
                    f"[OpenSearch] Bulk indexed {success} chunk docs for user={user_id}, doc_id={doc_id}"
                )
            except Exception as exc:
                print(
                    f"[OpenSearch] Bulk error (user={user_id} doc_id={doc_id}): {exc}"
                )
            actions = []

    # leftover
    if actions:
        try:
            success, errors = bulk(os_client, actions)
            print(
                f"[OpenSearch] Bulk indexed {success} chunk docs for user={user_id}, doc_id={doc_id}"
            )
        except Exception as exc:
            print(f"[OpenSearch] Bulk error (user={user_id} doc_id={doc_id}): {exc}")


###############################################################################
# Endpoint: Upload & Store
###############################################################################
@app.post("/upload_text")
async def upload_text(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    1) Derive doc_id from filename + timestamp
    2) Save file => uploads/{user_id}/doc_id.ext
    3) Read & chunk text
    4) Embeddings => user-specific index, doc_id repeated for all chunks
    """

    # Validate the filename
    if not file.filename.strip():
        raise HTTPException(status_code=400, detail="No valid filename provided.")

    if not user_id.strip():
        raise HTTPException(status_code=400, detail="No valid user-id provided.")

    # Derive doc_id from name + timestamp
    name_stem = pathlib.Path(file.filename).stem  # "report" from "report.txt"
    now_ts = int(time.time())
    doc_id = f"{name_stem}_{now_ts}"  # ex: "report_1692300000"

    # Make user-specific folder
    user_folder = os.path.join(BASE_UPLOAD_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    # Build final filename
    extension = pathlib.Path(file.filename).suffix
    final_filename = f"{doc_id}{extension}"  # "report_1692300000.txt"
    final_path = os.path.join(user_folder, final_filename)

    # Save file
    try:
        with open(final_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save error: {exc}")

    # Read file text
    try:
        with open(final_path, "r", encoding="utf-8") as f:
            contents = f.read()
    except UnicodeDecodeError:
        with open(final_path, "r", encoding="latin-1") as f:
            contents = f.read()

    if not contents.strip():
        raise HTTPException(
            status_code=400, detail="File is empty or contains no text."
        )

    # Chunk text
    chunks = chunk_text(contents, CHUNK_SIZE)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks generated.")

    # Embed
    embeddings = await embed_texts_in_batches(chunks)
    if embeddings.shape[0] != len(chunks):
        raise HTTPException(
            status_code=500, detail="Mismatch in chunk vs embedding counts!"
        )

    # Index
    bulk_index_embeddings(user_id, doc_id, embeddings, chunks)

    return {
        "user_id": user_id,
        "doc_id": doc_id,
        "saved_file": final_path,
        "num_chunks": len(chunks),
        "message": f"Uploaded '{file.filename}' & embedded for user='{user_id}' doc_id='{doc_id}'.",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9001, reload=False)
