"""LLM provider adapters for MedTrace fact extraction (Stage 4).

Providers are intentionally kept out of the core ``ai`` package exports:
the core schema/validation layer must remain usable without any LLM SDK::

    from ai import extract_pdf_text, chunk_extraction, extract_facts
    from ai.providers import GeminiFactExtractionProvider
"""

from .gemini_provider import DEFAULT_MODEL, GeminiFactExtractionProvider, GeminiProviderError

__all__ = [
    "DEFAULT_MODEL",
    "GeminiFactExtractionProvider",
    "GeminiProviderError",
]
