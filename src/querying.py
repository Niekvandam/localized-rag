# querying.py
import logging
from .vectorstore_utils import check_chroma_collection  # Import the helper
from llama_index.core.indices import VectorStoreIndex
logger = logging.getLogger(__name__)

async def query_llamaindex_rag(index: VectorStoreIndex, query_text):
    """
    Queries the LlamaIndex RAG system and returns the response.
    Logs extra debugging information about the underlying Chroma collection.
    """
    logger.info(f"Querying LlamaIndex RAG system with query: '{query_text}'")
    
    # Check the underlying vector store (Chroma) contents.
    if hasattr(index, "vector_store"):
        check_chroma_collection(index.vector_store)
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            node_postprocessors=[
                ]
        )
        response = await query_engine.aquery(query_text)
        logger.debug(f"Response source nodes: {response.source_nodes}")
        return response
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return None
