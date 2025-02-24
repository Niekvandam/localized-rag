# indexing.py
import os
import logging
from llama_index.core import VectorStoreIndex

from .config import MANIFEST_FILE

logger = logging.getLogger(__name__)

def index_nodes_llamaindex(index, embedding_model, nodes):
    """
    Indexes pre-processed nodes into the vector store using LlamaIndex.
    """
    logger.debug("Starting index_nodes_llamaindex.")
    if not nodes:
        logger.warning("No nodes provided for indexing; skipping.")
        return index

    try:
        logger.debug(f"Indexing {len(nodes)} nodes.")
        # Log first few node IDs for debugging.
        for i, node in enumerate(nodes[:5]):
            logger.debug(f"Node {i+1}: {node.node_id}")
        index.insert_nodes(nodes, embed_model=embedding_model)
        logger.info(f"Successfully indexed {len(nodes)} nodes.")

        # Optionally verify node count in the vector store (for ChromaDB).
        vector_store = index.vector_store
        if hasattr(vector_store, '_collection'):
            node_count = vector_store._collection.count()
            logger.info(f"Vector store now contains {node_count} nodes.")
        else:
            logger.warning("Vector store does not support node count retrieval.")
    except Exception as e:
        logger.error(f"Error during node indexing: {e}")
    logger.debug("Exiting index_nodes_llamaindex.")
    return index

def load_existing_index(vector_store, manifest_file=MANIFEST_FILE):
    """Loads an existing LlamaIndex index from disk if it exists, otherwise returns None.
    Forces re-indexing if manifest exists but index directory is missing.
    """
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index
    except Exception as e:
        if os.path.exists(manifest_file):
            logger.info(f"Couldnt load index, but manifest file '{manifest_file}' exists.")
            logger.info("Manifest file indicates documents were previously indexed. Forcing re-indexing.")
            return None
        else:
            logger.info("Starting from scratch.")
            return None