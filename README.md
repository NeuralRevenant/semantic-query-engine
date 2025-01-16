# **Semantic Query Engine**

A semantic text indexing and retrieval system designed for large-scale datasets in domains like healthcare, using state-of-the-art embedding models and scalable vector search technologies.

---

## **Key Features**
- **Semantic Search**: Leverages OpenAI's `text-embedding-ada-002` for high-dimensional embeddings of text data.
- **Retrieval-Augmented Generation (RAG)**: Combines semantic retrieval with GPT-4o for precise, context-aware results.
- **Hybrid Search System**: 
  - **FAISS**: In-memory, real-time Approximate Nearest Neighbor (ANN) search.
  - **Elasticsearch**: Scalable, persistent vector search with `dense_vector` support and `cosine` similarity.
- **REST API Integration**: Offers endpoints for semantic search and natural language query answering.
- **Dynamic Data Synchronization**: Replicates data between FAISS and Elasticsearch in the background, ensuring both systems are kept up-to-date without blocking query execution.

---

## **How It Works**
1. **Embedding Generation**:
   - Text is converted into high-dimensional vector representations using OpenAI's embedding model.
2. **Indexing**:
   - FAISS and Elasticsearch indexes are populated with these embeddings for efficient search.
3. **Search**:
   - Users can search via:
     - **FAISS**: Fast in-memory ANN search.
     - **Elasticsearch**: Persistent and scalable search.
     - **Combined**: Hybrid search with configurable prioritization.
4. **Query Augmentation**:
   - Query results are passed to GPT-4o for natural language response generation.

---

## **Endpoints**

### **1. Semantic Search with FAISS**
- **Endpoint**: `/search/faiss`
- **Method**: `POST`
- **Request**:
  ```json
  {
      "query": "Find all articles about cancer treatment.",
      "top_k": 3
  }
  ```
- **Response**:
  ```json
  {
      "query": "Find all articles about cancer treatment.",
      "top_k": 3,
      "results": [
          {"text": "Article 1 text", "distance": 0.123},
          {"text": "Article 2 text", "distance": 0.456}
      ]
  }
  ```

### **2. Semantic Search with Elasticsearch**
- **Endpoint**: `/search/elasticsearch`
- **Method**: `POST`
- **Request**: Same as `/search/faiss`.
- **Response**: Includes `score` instead of `distance`.

### **3. Combined Search**
- **Endpoint**: `/search/combined`
- **Method**: `POST`
- **Request**: Same as `/search/faiss`.
- **Response**: Combines results from FAISS and Elasticsearch.

### **4. Retrieval-Augmented Query Answering**
- **Endpoint**: `/ask`
- **Method**: `POST`
- **Request**:
  ```json
  {
      "query": "What are the side effects of cancer treatment?",
      "top_k": 3
  }
  ```
- **Response**:
  ```json
  {
      "query": "What are the side effects of cancer treatment?",
      "answer": "Common side effects include nausea, fatigue, and hair loss."
  }
  ```

---

## **Setup and Deployment**

### **Prerequisites**
1. **Python 3.8+**
2. **Dependencies**:
   - `fastapi`
   - `uvicorn`
   - `openai`
   - `faiss`
   - `elasticsearch`

3. **Environment Variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `ELASTIC_API_KEY`: Elasticsearch API key.

### **Run the Server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## **Code Highlights**

### **Background Replication**
The system ensures data consistency between FAISS and Elasticsearch:
- Replication tasks are triggered automatically when one index has data, and the other is empty.
- Tasks are scheduled in the background using `BackgroundTasks`.

### **Dynamic Data Handling**
- When one index is partially populated while replication, queries are restricted to the fully populated index to avoid incomplete results.

### **Context-Aware Querying**
- Combines semantic search results with GPT-4o to generate context-aware natural language answers.

---

## **Scalability**
1. **FAISS**:
   - Ideal for small to medium datasets.
   - Fast and efficient in-memory searches.

2. **Elasticsearch**:
   - Scalable to millions of documents.
   - Persistent storage with vector similarity search.

3. **Combined Search**:
   - Merges the benefits of both systems for flexible use cases.

---

## **Future Improvements**
- Add support for **Pinecone** as an optional vector database.
- Extend to multi-modal search (example: images + text).
- Integrate streaming capabilities for real-time updates to indices and periodic index updates as the data changes.

---

- Open for contributions! Feel free to submit issues or pull requests.
