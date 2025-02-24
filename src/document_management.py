# document_management.py
import os
import logging
from typing import List, Dict, Tuple, Optional

from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from .config import DOCUMENTS_DIR
from .manifest_utils import load_manifest, save_manifest  # Use centralized manifest handling

logger = logging.getLogger(__name__)

def load_documents_llamaindex(pdf_dir=".", file_paths=[]) -> List[Document]:
    """Loads PDF documents from a directory using LlamaIndex.
       If file_paths is provided, only those files are loaded.
    """
    if file_paths:
        logger.debug(f"Loading documents from provided file paths: {file_paths}")
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    else:
        logger.debug(f"Loading documents from directory: {pdf_dir}")
        documents = SimpleDirectoryReader(input_dir=pdf_dir, required_exts=[".pdf"]).load_data()
    logger.debug(f"Documents loaded: {len(documents)}")
    if documents:
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

async def sync_documents_with_vector_store(vector_store, embedding_model, ingestion_pipeline, 
                                             documents_dir: str, manifest_file: str, 
                                             existing_index=None) -> Optional[VectorStoreIndex]:
    """
    Synchronizes documents in `documents_dir` with the vector store.
    If only new documents are detected, they are inserted incrementally.
    Full re-indexing is triggered only if documents were updated or deleted.
    """
    logger.info("Starting document synchronization with vector store.")
    current_document_state = get_document_state(documents_dir)
    manifest_data = load_manifest(manifest_file)

    docs_to_add = []
    docs_to_update = []
    filenames_to_delete = []
    index_updated = False
    index_to_return = existing_index

    # Determine which documents are new, updated, or deleted.
    for filename, (last_modified, file_size) in current_document_state.items():
        if filename not in manifest_data:
            logger.info(f"New document detected: {filename}")
            docs_to_add.append(os.path.join(documents_dir, filename))
        else:
            old_entry = manifest_data.get(filename, {})
            old_last_modified = old_entry.get('last_modified')
            old_file_size = old_entry.get('file_size')
            if old_last_modified is None or old_file_size is None:
                logger.warning(f"Incomplete manifest entry for {filename}. Marking for update.")
                docs_to_update.append(os.path.join(documents_dir, filename))
            elif last_modified > old_last_modified or file_size != old_file_size:
                logger.info(f"Document updated: {filename}")
                docs_to_update.append(os.path.join(documents_dir, filename))
    # Identify deleted documents.
    for filename in manifest_data:
        if filename not in current_document_state and filename.endswith(".pdf"):
            logger.info(f"Document deleted: {filename}")
            filenames_to_delete.append(filename)

    # Case 1: If there are any updates or deletions, re-index all documents.
    if docs_to_update or filenames_to_delete:
        logger.info("Updates or deletions detected; performing full re-indexing.")
        all_documents = load_documents_llamaindex(pdf_dir=documents_dir)
        if all_documents:
            try:
                index_to_return = VectorStoreIndex.from_documents(
                    all_documents,
                    vector_store=vector_store,
                    embed_model=embedding_model,
                )
                nodes = await ingestion_pipeline.arun(
                    documents=all_documents, in_place=True, show_progress=True
                )
                from indexing import index_nodes_llamaindex
                index_nodes_llamaindex(index_to_return, embedding_model, nodes)
                logger.info("Full re-indexing complete for updated/deleted documents.")
                index_updated = True
            except Exception as e:
                logger.error(f"Error during re-indexing: {e}")
        else:
            logger.warning("No documents available for re-indexing.")
    # Case 2: If there are only new documents, insert them incrementally.
    elif docs_to_add:
        logger.info("Only new documents detected; inserting incrementally.")
        # Assume load_documents_llamaindex can be given a list of file paths.
        new_documents = load_documents_llamaindex(file_paths=docs_to_add)
        if new_documents:
            if index_to_return is None:
                logger.info("No existing index found; creating a new index from new documents.")
                index_to_return = VectorStoreIndex.from_documents(
                    new_documents,
                    vector_store=vector_store,
                    embed_model=embedding_model,
                )
            else:
                try:
                    new_nodes = await ingestion_pipeline.arun(
                        documents=new_documents, in_place=True, show_progress=True
                    )
                    from indexing import index_nodes_llamaindex
                    # Append new nodes into the existing index.
                    index_nodes_llamaindex(index_to_return, embedding_model, new_nodes)
                    logger.info(f"Incrementally indexed {len(new_documents)} new documents.")
                    index_updated = True
                except Exception as e:
                    logger.error(f"Error inserting new documents: {e}")
        else:
            logger.warning("No new documents were loaded despite detection.")
    else:
        logger.info("No changes detected; index remains unchanged.")

    # Always update the manifest.
    updated_manifest_data = {
        filename: {"last_modified": lm, "file_size": sz}
        for filename, (lm, sz) in current_document_state.items()
    }
    save_manifest(updated_manifest_data, manifest_file)
    return index_to_return
