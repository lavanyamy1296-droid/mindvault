from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
