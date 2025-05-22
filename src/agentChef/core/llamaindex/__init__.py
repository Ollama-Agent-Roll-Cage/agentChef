"""LlamaIndex integration for natural language DataFrame querying."""

try:
    import pandas as pd
    from llama_index.experimental.query_engine import PandasQueryEngine
    HAS_QUERY_ENGINE = True
except ImportError:
    HAS_QUERY_ENGINE = False
    import logging
    logging.warning("LlamaIndex not installed. Install with 'pip install llama-index[experimental]'")

from .pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration

__all__ = [
    'PandasQueryIntegration', 
    'OllamaLlamaIndexIntegration',
    'HAS_LLAMA_INDEX',
    'PandasQueryEngine'
]
