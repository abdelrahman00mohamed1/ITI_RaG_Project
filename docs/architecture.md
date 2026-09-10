# Architecture & Technical Design

## End-to-End Pipeline

1. **Document Ingestion & Auditing**:
   - Source: data/raw/harrypotter.pdf (3,623 pages).
   - Tool: PyMuPDF (itz).
   - Audit: Digital native text streams across all chapters, confirming zero OCR dependency.

2. **Text Chunking Strategy**:
   - Window size: 700 characters (~150 words).
   - Overlap: 100 characters (~20 words).
   - Metadata: ook_name, page_number, chunk_id.

3. **Dense Embeddings**:
   - Model: intfloat/multilingual-e5-large (1,024 dimensions).
   - Prefixing: passage:  for indexing, query:  for retrieval.

4. **Vector Database**:
   - Storage: Local persistent **ChromaDB** (data/vector_store/).
   - Distance metric: Cosine similarity.

5. **FastAPI Backend**:
   - Endpoints: GET /health and POST /query.
   - Lifespan initialization: Models and vector store are loaded once at startup.
   - Pydantic v2 schemas: QueryRequest (supporting query and question) and QueryResponse.

6. **Query Routing & Generation**:
   - Intent Router: Groq (llama-3.3-70b-versatile) classifying into 
etrieve, chitchat, or off-topic.
   - Grounded RAG Generation: Google Gemini (gemini-1.5-flash) with strict anti-hallucination instruction.

7. **Frontend Experience**:
   - Streamlit chat application (rontend/app.py).
   - Custom responsive web UI (rontend/web/index.html).
