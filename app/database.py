import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from .get_embedding_function import get_embedding_function
from PyPDF2 import PdfReader

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    # Add metadata from PDFs
    for doc in docs:
        pdf_path = doc.metadata.get("source")
        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                meta = reader.metadata
                if meta:
                    doc.metadata.update({k[1:]: str(v) for k, v in meta.items()})  # strip leading '/'
        except Exception as e:
            print(f"Error loading metadata for {pdf_path}: {e}")
    return docs

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    chunk_idx = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_id = f"{source}:{page}"

        if current_id == last_page_id:
            chunk_idx += 1
        else:
            chunk_idx = 0

        chunk.metadata["id"] = f"{current_id}:{chunk_idx}"
        last_page_id = current_id

    return chunks

def add_to_chroma(chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks = calculate_chunk_ids(chunks)

    existing_ids = set(db.get()["ids"])
    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        db.persist()

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def populate(reset=False):
    if reset:
        clear_database()
    docs = load_documents()
    chunks = split_documents(docs)
    add_to_chroma(chunks)
