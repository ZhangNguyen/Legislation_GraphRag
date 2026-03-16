from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()
class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY","")
    openai_model: str = os.getenv("OPENAI_MODEL","gpt-4o-mini")
    openai_embeding_model: str = os.getenv("OPENAI_EMBEDING_MODEL","text-embedding-3-small")

    #Qdrant
    qdrant_url: str = os.getenv("QRANT_URL","http://localhost:6333")
    qdrant_api_key: str = os.getenv("QRANT_API_KEY","")
    qdrant_collection: str = os.getenv("QRANT_COLLECTION","legal_chunks")

    top_k: int = int(os.getenv("TOP_K",10))

settings = Settings()