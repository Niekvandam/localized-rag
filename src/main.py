# main.py
import logging
import sys
from .config import (
    DOCUMENTS_DIR,
    setup_llama_chroma_db,
    setup_llama_ollama_embedding,
    setup_ingestion_pipeline,
)
from document_management import sync_documents_with_vector_store
from indexing import load_existing_index
from querying import query_llamaindex_rag
from llama_index.core.storage.chat_store import SimpleChatStore
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

CHATSTORE_PERSIST_PATH = "chat_store.json"
def main():
    try:
        pdf_directory = DOCUMENTS_DIR

        # Setup persistent ChromaDB vector store and get the persistent client.
        vector_store, chroma_client = setup_llama_chroma_db()

        # Setup Ollama embedding model.
        ollama_embed_model = setup_llama_ollama_embedding()

        # Setup Ingestion Pipeline.
        ingestion_pipeline = setup_ingestion_pipeline()

        # Load existing index if available.
        index = load_existing_index(vector_store)
        index = sync_documents_with_vector_store(
            vector_store,
            ollama_embed_model,
            ingestion_pipeline,
            pdf_directory,
            "manifest.json",
            existing_index=index
        )
        try:
            chat_store = SimpleChatStore.from_persist_path(CHATSTORE_PERSIST_PATH)
            logger.info("Loaded persisted chat store.")
        except Exception as e:
            logger.info("No persisted chat store found; creating a new one.")
            chat_store = SimpleChatStore()

        if index is not None:
            # Persist the Llama index.
            index_load_query = load_existing_index(vector_store)
            index_load_query = index
            logger.info(index.vector_store)
            if index_load_query:
                while True:
                    user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
                    if user_query.lower() == 'exit':
                        break
                    response = query_llamaindex_rag(index_load_query, user_query)
                    if response:
                        logger.info(f"Query Response:\n{response}")
                        logger.info(f"Used sources: {response.source_nodes}")
                    else:
                        logger.error("Query failed; see previous logs for details.")
            else:
                logger.error("Failed to load index after synchronization.")
        else:
            logger.warning("No index available for querying. Exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main(): {e}")

if __name__ == "__main__":
    main()
