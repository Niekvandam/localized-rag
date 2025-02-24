# main_fastapi.py
import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .config import (
    DOCUMENTS_DIR,
    MANIFEST_FILE,
    setup_llama_chroma_db,
    setup_llama_ollama_embedding,
    setup_ingestion_pipeline,
    APP_CONFIG,
    save_app_config,
)
from .document_management import get_document_state, sync_documents_with_vector_store
from .indexing import load_existing_index
from .querying import query_llamaindex_rag

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LlamaIndex FastAPI Service")

# Global variables to hold our application state.
vector_store = None
chroma_client = None
embedding_model = None
ingestion_pipeline = None
index = None

@app.on_event("startup")
async def startup_event():
    """
    Initializes the vector store, embedding model, ingestion pipeline,
    loads (and synchronizes) the index at startup.
    """
    global vector_store, chroma_client, embedding_model, ingestion_pipeline, index
    vector_store, chroma_client = setup_llama_chroma_db()
    embedding_model = setup_llama_ollama_embedding()
    ingestion_pipeline = setup_ingestion_pipeline()
    index = load_existing_index(vector_store)
    index = await sync_documents_with_vector_store(
        vector_store,
        embedding_model,
        ingestion_pipeline,
        DOCUMENTS_DIR,
        MANIFEST_FILE,
        existing_index=index,
    )
    logger.info("Startup complete. Index synchronized.")

# -------------------------
# Documents Endpoints
# -------------------------

@app.get("/documents")
def get_documents():
    document_state = get_document_state(DOCUMENTS_DIR)
    return document_state

@app.post("/documents")
async def add_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_location = os.path.join(DOCUMENTS_DIR, file.filename)
    try:
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save the file.")
    global index
    index = await sync_documents_with_vector_store(
        vector_store,
        embedding_model,
        ingestion_pipeline,
        DOCUMENTS_DIR,
        MANIFEST_FILE,
        existing_index=index,
    )
    return {"message": f"File '{file.filename}' uploaded and indexed successfully."}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        os.remove(file_path)
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete the file.")
    global index
    index = await sync_documents_with_vector_store(
        vector_store,
        embedding_model,
        ingestion_pipeline,
        DOCUMENTS_DIR,
        MANIFEST_FILE,
        existing_index=index,
    )
    return {"message": f"File '{filename}' deleted and index updated."}

# -------------------------
# Chat Endpoint
# -------------------------

@app.post("/chat")
async def chat(query: dict):
    query_text = query.get("query")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")
    global index
    if index is None:
        raise HTTPException(status_code=500, detail="Index not available for querying.")
    response = await query_llamaindex_rag(index, query_text)
    if response is None:
        raise HTTPException(status_code=500, detail="Query failed.")
    return {"response": str(response), "sources": str(response.source_nodes)}

# -------------------------
# Configuration Endpoints
# -------------------------

@app.get("/config")
def get_config():
    return APP_CONFIG

@app.put("/config")
def update_config(new_config: dict):
    """
    Updates the global configuration, persists the changes to disk,
    and returns the updated configuration.
    """
    APP_CONFIG.update(new_config)
    save_app_config(APP_CONFIG)
    return {"message": "Configuration updated successfully.", "config": APP_CONFIG}
