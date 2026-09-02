# rag_config.py
# Compatibility shim so document_store.py from the RAG project can import its config.
# document_store.py does: from rag_config import get_document_config, get_rag_config

from academic_config import get_rag_config, get_document_config, get_llm_config, get_thinking_config
