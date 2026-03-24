from __future__ import annotations

import logging
from threading import Lock
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.app.settings import settings

logger = logging.getLogger(__name__)

_llm: Optional[ChatOpenAI] = None
_embeddings: Optional[OpenAIEmbeddings] = None
_client_lock = Lock()


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is not None:
        return _llm
    if not settings.openai_api_key:
        raise ValueError("Missing api key OPENAI")
    with _client_lock:
        if _llm is None:
            _llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.2,
            )
    return _llm


def get_embedings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if not settings.openai_api_key:
        raise ValueError("Missing api key OPENAI")
    with _client_lock:
        if _embeddings is None:
            _embeddings = OpenAIEmbeddings(
                model=settings.openai_embed_model,
                api_key=settings.openai_api_key,
            )
    return _embeddings


def warmup_openai_clients() -> None:
    try:
        get_llm()
        get_embedings()
    except Exception as exc:
        logger.warning("OpenAI client warmup failed: %s", exc)


def get_embeding_dim(emb: OpenAIEmbeddings) -> int:
    v = emb.embed_query("dim_check")
    return len(v)
