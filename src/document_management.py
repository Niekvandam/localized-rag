# document_management.py
import os
import logging
from typing import List, Dict, Tuple, Optional

from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from .config import DOCUMENTS_DIR
from .manifest_utils import load_manifest, save_manifest  # Use centralized manifest handling

logger = logging.getLogger(__name__)

def load_documents_llamaindex(pdf_dir=".") -> List[Document]:
    """Loads PDF documents from a directory using LlamaIndex."""
    logger.debug(f"Loading documents from directory: {pdf_dir}")
    documents = SimpleDirectoryReader(input_dir=pdf_dir, required_exts=[".pdf"]).load_data()
    logger.debug(f"Documents loaded: {len(documents)}")
    if documents: # Log the first few characters of the first document
        logger.debug(f"First document content snippet: {documents[0].text[:100]}...")
    return documents

def get_document_state(documents_dir: str) -> Dict[str, Tuple[float, int]]:
    """
    Returns a dictionary representing the state of documents in the given directory.
    Keys are filenames, values are tuples of (last modified timestamp, file size).
    """
    document_state = {}
    try:
        for filename in os.listdir(documents_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(documents_dir, filename)
                if os.path.isfile(filepath):
                    last_modified = os.path.getmtime(filepath)
                    file_size = os.path.getsize(filepath)
                    document_state[filename] = (last_modified, file_size)
    except Exception as e:
        logger.error(f"Error accessing documents directory '{documents_dir}': {e}")
    logger.debug(f"Document state: {document_state}")
    return document_state

async def sync_documents_with_vector_store(vector_store, embedding_model, ingestion_pipeline, documents_dir: str, manifest_file: str, existing_index=None) -> Optional[VectorStoreIndex]:
    """
    Synchronizes documents in the specified directory with the vector store.
    Detects new, updated, or deleted documents and re-indexes as needed.
    Returns the updated index (or None if not available).
    """
    logger.info("Starting document synchronization with vector store.")
    current_document_state = get_document_state(documents_dir)
    manifest_data = load_manifest(manifest_file)

    docs_to_add = []
    docs_to_update = []
    filenames_to_delete = []
    index_updated = False
    index_to_return = existing_index

    # If no existing index but a manifest exists, force re-indexing.
    if existing_index is None and manifest_data:
        logger.info("Forcing re-index: No existing index loaded, but manifest exists.")
        docs_to_add = [os.path.join(documents_dir, filename)
                       for filename in current_document_state if filename.endswith(".pdf")]
        if not docs_to_add:
            logger.warning("Manifest exists but no documents found. Creating empty index.")
            index_to_return = VectorStoreIndex(vector_store=vector_store, embed_model=embedding_model)
            index_updated = True
    else:
        # Detect changes based on modification time or file size.
        for filename, (last_modified, file_size) in current_document_state.items():
            if filename not in manifest_data:
                logger.info(f"New document detected: {filename}")
                docs_to_add.append(os.path.join(documents_dir, filename))
            else:
                old_entry = manifest_data.get(filename, {})
                old_last_modified = old_entry.get('last_modified')
                old_file_size = old_entry.get('file_size')
                if old_last_modified is None or old_file_size is None:
                    logger.warning(f"Incomplete manifest entry for {filename}. Re-indexing.")
                    docs_to_update.append(os.path.join(documents_dir, filename))
                elif last_modified > old_last_modified or file_size != old_file_size:
                    logger.info(f"Document updated: {filename}")
                    docs_to_update.append(os.path.join(documents_dir, filename))

        # Identify deleted documents.
        for filename in manifest_data:
            if filename not in current_document_state and filename.endswith(".pdf"):
                logger.info(f"Document deleted: {filename}")
                filenames_to_delete.append(filename)

    # Process new documents.
    if docs_to_add:
        logger.info("Adding new documents with preprocessing...")
        new_documents = load_documents_llamaindex(pdf_dir=documents_dir)
        if new_documents:
            if index_to_return is None:
                logger.info("Creating a new index for new documents.")
                index_to_return = VectorStoreIndex.from_documents(
                    new_documents,
                    vector_store=vector_store,
                    embed_model=embedding_model,
                )
            try:
                nodes = await ingestion_pipeline.arun(documents=new_documents, in_place=True, show_progress=True)
                from indexing import index_nodes_llamaindex  # Local import to avoid circular dependency
                index_nodes_llamaindex(index_to_return, embedding_model, nodes)
                logger.info(f"Indexed {len(docs_to_add)} new documents.")
                index_updated = True
            except Exception as e:
                logger.error(f"Error indexing new documents: {e}")
        else:
            logger.warning("No new documents were loaded despite detection.")

    # Process updates and deletions by re-indexing all documents.
    if docs_to_update or filenames_to_delete:
        logger.info("Re-indexing due to updates or deletions.")
        all_documents = load_documents_llamaindex(pdf_dir=documents_dir)
        if all_documents:
            try:
                index_to_return = VectorStoreIndex.from_documents(
                    all_documents,
                    vector_store=vector_store,
                    embed_model=embedding_model,
                )
                nodes = await ingestion_pipeline.arun(documents=all_documents, in_place=True, show_progress=True)
                from indexing import index_nodes_llamaindex
                index_nodes_llamaindex(index_to_return, embedding_model, nodes)
                logger.info("Re-indexing complete for updated/deleted documents.")
                index_updated = True
            except Exception as e:
                logger.error(f"Error during re-indexing: {e}")
        else:
            logger.warning("No documents available for re-indexing after updates/deletions.")

    # Update manifest regardless of changes.
    updated_manifest_data = {
        filename: {"last_modified": lm, "file_size": sz}
        for filename, (lm, sz) in current_document_state.items()
    }
    save_manifest(updated_manifest_data, manifest_file)
    return index_to_return
