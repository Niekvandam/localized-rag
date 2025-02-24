# vectorstore_utils.py
import logging

logger = logging.getLogger(__name__)

def check_chroma_collection(vector_store):
    """
    Checks and logs the number of nodes in the underlying Chroma collection,
    as well as a sample of node IDs.
    """
    if hasattr(vector_store, "_collection"):
        try:
            node_count = vector_store._collection.count()
            logger.info(f"Chroma collection contains {node_count} nodes.")
            # Optionally, list a few node IDs if supported:
            # For example, if _collection has a 'get' or 'peek' method.
            sample = vector_store._collection.get()  # Adjust based on your chroma API
            logger.debug(f"Sample from collection: {sample}")
        except Exception as e:
            logger.error(f"Error checking Chroma collection: {e}")
    else:
        logger.warning("Vector store does not expose an underlying _collection attribute.")
