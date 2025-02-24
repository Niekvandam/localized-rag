# manifest_utils.py
import json
import logging
import os

logger = logging.getLogger(__name__)

def load_manifest(manifest_file: str) -> dict:
    """
    Loads the manifest file if it exists, returning an empty dict otherwise.
    Handles errors gracefully.
    """
    try:
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                logger.debug(f"Loaded manifest from '{manifest_file}': {manifest}")
                return manifest
        else:
            logger.debug(f"Manifest file '{manifest_file}' not found. Returning empty manifest.")
            return {}
    except Exception as e:
        logger.error(f"Error loading manifest file '{manifest_file}': {e}")
        return {}

def save_manifest(manifest: dict, manifest_file: str) -> None:
    """
    Saves the manifest to a JSON file and handles errors gracefully.
    """
    try:
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=4)
        logger.debug(f"Saved manifest to '{manifest_file}': {manifest}")
    except Exception as e:
        logger.error(f"Error saving manifest file '{manifest_file}': {e}")
