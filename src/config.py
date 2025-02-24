# config.py
import os
import json
import logging
import sys
import chromadb

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration persistence file
APP_CONFIG_FILE = "app_config.json"

# Default configuration settings.
DEFAULT_APP_CONFIG = {
    "llm": {"model": "llama3.2", "request_timeout": 1500},
    "embed_model": {"model_name": "nomic-embed-text", "timeout": 150},
    "documents_dir": "data/",
    "manifest_file": "manifest.json",
}

def load_app_config():
    """Loads the configuration from disk, or returns the default configuration if the file doesn't exist."""
    if os.path.exists(APP_CONFIG_FILE):
        try:
            with open(APP_CONFIG_FILE, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {APP_CONFIG_FILE}")
                return config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    logger.info("Using default configuration.")
    return DEFAULT_APP_CONFIG.copy()

def save_app_config(config):
    """Saves the configuration to disk."""
    try:
        with open(APP_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {APP_CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving config file: {e}")

# Load the configuration at module load time.
APP_CONFIG = load_app_config()

# Use the configuration to set up underlying settings.
Settings.llm = Ollama(model=APP_CONFIG["llm"]["model"], request_timeout=APP_CONFIG["llm"]["request_timeout"])
Settings.embed_model = OllamaEmbedding(model_name=APP_CONFIG["embed_model"]["model_name"], timeout=APP_CONFIG["embed_model"]["timeout"])

DOCUMENTS_DIR = APP_CONFIG["documents_dir"]
MANIFEST_FILE = APP_CONFIG["manifest_file"]

def setup_llama_chroma_db():
    """
    Sets up a persistent ChromaDB vector store and returns both the vector store
    and the underlying persistent client so that it can be explicitly persisted.
    """
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.get_or_create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    logger.debug("ChromaDB vector store setup complete.")
    return vector_store, chroma_client

def setup_llama_ollama_embedding():
    """Sets up Ollama as the embedding model in LlamaIndex."""
    logger.debug("Setting up Ollama embedding model.")
    ollama_embedding = OllamaEmbedding(model_name=APP_CONFIG["embed_model"]["model_name"],
                                       timeout=APP_CONFIG["embed_model"]["timeout"])
    logger.debug("Ollama embedding model setup complete.")
    return ollama_embedding

def setup_ingestion_pipeline():
    """Sets up LlamaIndex ingestion pipeline with metadata extractors and text splitter."""
    logger.debug("Setting up LlamaIndex ingestion pipeline.")
    from llama_index.core.extractors import TitleExtractor, SummaryExtractor, QuestionsAnsweredExtractor
    from llama_index.core.text_splitter import TokenTextSplitter
    from llama_index.core.ingestion import IngestionPipeline

    text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=128)
    title_extractor = TitleExtractor(nodes_batch_size=5)
    summary_extractor = SummaryExtractor(summaries=["self"], nodes_batch_size=5)
    questions_answered = QuestionsAnsweredExtractor(questions=5)
    pipeline = IngestionPipeline(
        transformations=[text_splitter, title_extractor, summary_extractor, questions_answered]
    )
    logger.debug("LlamaIndex ingestion pipeline setup complete.")
    return pipeline
