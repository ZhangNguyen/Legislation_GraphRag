from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.app.settings import settings

def get_llm() -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("Missing api key OPENAI")
    return ChatOpenAI(model=settings.openai_model,api_key=settings.openai_api_key, temperature=0.2)

def get_embedings() -> OpenAIEmbeddings:
    if not settings.openai_api_key:
        raise ValueError("Missing api key OPENAI")
    return OpenAIEmbeddings(model=settings.openai_embeding_model,api_key=settings.openai_api_key)

def get_embeding_dim(emb: OpenAIEmbeddings) -> int:
    # gọi để biết dimension
    v = emb.embed_query("dim_check")
    return len(v)

