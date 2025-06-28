
# SE-RAG-Project

A Retrieval‑Augmented Generation (RAG) web service built with FastAPI, Chroma vector store, and Ollama LLM. Users can upload PDFs, index their content into a local Chroma database, and query the system for context‑aware answers.

---

## Table of Contents

1. Features
2. Prerequisites
3. Installation
4. Configuration
5. Project Structure
6. Running the Application
7. API Endpoints
8. Testing

---

## Features

* **PDF Upload**
  Upload PDF files via HTTP POST and save them to a local directory.

* **Metadata Extraction**
  Reads PDF metadata (author, title, etc.) with PyPDF2 and attaches it to each document chunk.

* **Text Splitting & Chunking**
  Splits loaded documents into overlapping text chunks using LangChain’s `RecursiveCharacterTextSplitter`.

* **Chroma Vector Store**
  Stores and persists embeddings in a local Chroma directory, supports incremental updates.

* **Embeddings**
  Generates embeddings with Sentence‑Transformers (`all-MiniLM-L6-v2`) via LangChain’s community integrations.

* **LLM Inference**
  Routes RAG prompts through Ollama’s local model (`deepseek-r1:1.5b`) for answer generation.

* **FastAPI Service**
  Exposes REST endpoints for uploading PDFs, populating the database, and querying the RAG pipeline.

* **Jinja2 Frontend**
  Serves a simple HTML interface for interactive use (file upload form and AJAX query).

* **Automated Tests**
  Unit tests for chunk‑ID assignment, document splitting, database clearing, embedding function, and RAG flow.

---

## Prerequisites

* Python 3.9 or higher
* Git
* (Optional) Docker for running Chroma or Ollama containers

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Disha-Savant/SE-RAG-Project.git
   cd SE-RAG-Project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux  
   venv\Scripts\activate        # Windows  
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Edit `app/config.py` (or override via environment variables) to adjust:

```python
CHROMA_PATH     = "chroma"                             # Directory for vector store  
DATA_PATH       = "data"                               # Upload directory for PDFs  
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
OLLAMA_MODEL    = "deepseek-r1:1.5b"                   # Ollama model tag  
```

---

## Project Structure

```
SE-RAG-Project/
├── app/
│   ├── __init__.py
│   ├── config.py                # Paths & model names
│   ├── database.py              # PDF loader, splitter, Chroma ingestion
│   ├── get_embedding_function.py
│   ├── rag.py                   # RAG query logic
│   ├── schemas.py               # Pydantic request/response models
│   ├── main.py                  # FastAPI app + routes
│   ├── static/                  # CSS, JS assets for frontend
│   └── templates/               # Jinja2 HTML templates
├── test_business_logic.py       # Unit tests for business logic
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Running the Application

1. **Start the FastAPI server**

   ```bash
   uvicorn app.main:app --reload
   ```

2. **Open the UI in your browser**
   Navigate to `http://localhost:8000/`

---

## API Endpoints

### POST `/upload/`

* **Description**: Upload a PDF file.
* **Request**: `multipart/form-data` with field `file`.
* **Response**: `{ "filename": "your_document.pdf" }`

### POST `/populate/`

* **Description**: Load all PDFs from `DATA_PATH`, split into chunks, and store embeddings in Chroma.
* **Query Parameter**: `reset` (boolean) – if true, clears existing Chroma directory first.
* **Response**: `{ "message": "Database populated." }`

### POST `/query/`

* **Description**: Ask a question against the indexed corpus.
* **Request Body**:

  ```json
  { "query": "Your question here" }
  ```
* **Response**:

  ```json
  {
    "response": "Generated answer …",
    "sources": [
      { "source": "data/doc.pdf", "page": 2, "id": "data/doc.pdf:2:0" },
      …
    ]
  }
  ```

---

## Testing

Execute unit tests with:

```bash
pytest test_business_logic.py
```

Test coverage includes:

* `calculate_chunk_ids` logic
* Document splitting
* Clearing the Chroma directory
* Embedding function instantiation
* Full RAG query flow (with mocked Chroma & Ollama)

---
