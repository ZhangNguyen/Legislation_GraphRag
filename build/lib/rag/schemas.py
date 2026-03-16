from pydantic import BaseModel,Field
from typing import Optional,Dict,Any

class LegalMetadata(BaseModel):
    law_name: str
    law_type: str
    year: int
    source: str
    article: Optional[str] = None
    chunk_id: str

class IngestChunk(BaseModel):
    text: str
    metadata: LegalMetadata

class ChatRequest(BaseModel):
    question: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None

class SourceItem(BaseModel):
    chunk_id: str
    law_name:str
    law_type: str
    year: int
    source: str
    article: Optional[str]
    snippet:str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]