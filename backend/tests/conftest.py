"""
Shared test setup: no network, no model downloads, no running Qdrant.
Environment overrides must be set before backend.app is imported, because
get_settings() is cached at first use.
"""

import os

os.environ["USE_RERANKER"] = "false"
os.environ["USE_HYDE"] = "false"
os.environ["OPENAI_API_KEY"] = ""
# Point at a port nothing listens on, so connect() fails fast and stays non-fatal
os.environ["QDRANT_HOST"] = "127.0.0.1"
os.environ["QDRANT_PORT"] = "1"
