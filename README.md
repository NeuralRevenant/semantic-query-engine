# **RAG-based Semantic Query Engine**

A Retrieval-Augmented Generation (RAG) pipeline that enables semantic search and query answering for medical documents. This system integrates **Ollama's Mistral-7B embeddings**, **OpenSearch for vector-based Approximate Nearest Neighbor (ANN) search**, and **Redis for query caching** to provide efficient and accurate responses to user queries.

---

## **Key Features**
- **Semantic Search**: Uses **Mistral-7B** embeddings via Ollama to generate high-dimensional text representations.
- **Retrieval-Augmented Generation (RAG)**: Combines OpenSearch-based semantic retrieval with Mistral-7B for intelligent query answering.
- **Efficient Caching**:
  - **Redis** stores previous queries and responses to optimize search speed.
  - Uses **cosine similarity** for cache lookups to avoid redundant computation.
- **Fast and Scalable Search**:
  - **OpenSearch HNSW ANN index** provides fast nearest-neighbor search for embeddings.
  - **Supports large-scale document indexing and retrieval.**
- **REST API Integration**: Exposes endpoints for search and question-answering over indexed medical texts.
- **Robust Text Processing**:
  - Cleans and chunks text before embedding.
  - Allows retrieval of multiple document chunks for context-aware responses.

---

## **How It Works**
1. **Embedding Generation**:
   - Input text is converted into embeddings using **Mistral-7B** via Ollama.
2. **Indexing**:
   - OpenSearch indexes document chunks with corresponding embeddings.
3. **Query Execution**:
   - Query embeddings are compared with cached results in **Redis**.
   - If not found, OpenSearch performs ANN-based retrieval.
   - Relevant documents are passed to Mistral-7B for final response generation.
   - The response is stored in Redis for future queries.

---

## **Endpoints**

### **1. Semantic Search with OpenSearch**
- **Endpoint**: `/search/opensearch`
- **Method**: `POST`
- **Request**:
  ```json
  {
      "query": "What are the symptoms of pneumonia?",
      "top_k": 3
  }
  ```
- **Response**:
  ```json
  {
      "query": "What are the symptoms of pneumonia?",
      "top_k": 3,
      "results": [
          {"doc_id": "PMC12345.txt", "text": "Pneumonia symptoms include cough, fever, and difficulty breathing.", "score": 0.87},
          {"doc_id": "PMC67890.txt", "text": "Symptoms can vary but commonly include fever and shortness of breath.", "score": 0.85}
      ]
  }
  ```

### **2. Retrieval-Augmented Query Answering**
- **Endpoint**: `/ask`
- **Method**: `POST`
- **Request**:
  ```json
  {
      "query": "How does pneumonia affect lung function?",
      "top_k": 3
  }
  ```
- **Response**:
  ```json
  {
      "query": "How does pneumonia affect lung function?",
      "answer": "Pneumonia leads to inflammation in the lungs, causing fluid buildup and reduced oxygen exchange."
  }
  ```

---

## **Setup and Deployment**

### **Prerequisites**
1. **Python 3.8+**
2. **Dependencies**:
   - `fastapi`
   - `uvicorn`
   - `requests`
   - `numpy`
   - `opensearch-py`
   - `redis`

3. **Environment Setup**:
   - **Ollama**: Running on `http://localhost:11434/api`
   - **OpenSearch**: Hosted locally or in a cloud environment
   - **Redis**: Running locally on port `6379`

### **Run the Server**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## **Code Highlights**

### **1. Redis-based Query Caching**
- Stores query embeddings and responses to optimize performance.
- Uses **cosine similarity** to identify previously asked queries and avoid redundant API calls.

### **2. OpenSearch ANN Indexing**
- Documents are indexed using **HNSW (Hierarchical Navigable Small World)** for fast vector retrieval.
- Search results are ranked based on similarity scores.

### **3. Dynamic Query Handling**
- Queries first check Redis cache for stored responses.
- If no cached response is found, OpenSearch retrieves relevant documents.
- Mistral-7B in **Ollama** generates final responses based on retrieved context.
- The new query-response pair is cached in Redis for future lookups.

---

## **Scalability & Performance**
- **OpenSearch** supports indexing **millions of documents** for scalable retrieval.
- **Redis caching** significantly reduces response latency.
- **Ollama embeddings** ensure accurate semantic understanding.

---

## **Future Improvements**
- Add support for **multi-modal retrieval** (text + images).
- Extend OpenSearch with **multi-hop document retrieval**.
- Optimize **query expansion techniques** for better recall.
- Implement **continuous learning** to refine embeddings over time.

---

Open for contributions! Feel free to submit issues or pull requests.

