from typing import Optional
from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint.
    Supports both 'query' and 'question' field names for compatibility.
    """
    query: Optional[str] = Field(
        default=None,
        description="The user's question or search query",
        examples=["Who is the Half-Blood Prince?"],
    )
    question: Optional[str] = Field(
        default=None,
        description="Alternative field name matching official graduation criteria",
        examples=["Who is the Half-Blood Prince?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Optional override for number of retrieved chunks",
        ge=1,
        le=10,
    )

    @model_validator(mode="after")
    def validate_query_or_question(self) -> "QueryRequest":
        q = self.query or self.question
        if not q or not q.strip():
            raise ValueError("Either 'query' or 'question' must be provided and non-empty.")
        self.query = q.strip()
        self.question = self.query
        return self


class Source(BaseModel):
    """Source citation metadata schema."""
    book_name: str = Field(description="Title of the Harry Potter book")
    page_number: int = Field(description="Page number from the original publication")
    score: float = Field(description="Similarity or relevance score")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier")
    content_snippet: Optional[str] = Field(default=None, description="Brief preview of chunk text")


class QueryResponse(BaseModel):
    """Response schema returned by the /query endpoint."""
    query: str = Field(description="The original user query")
    route: str = Field(description="Classified route: 'retrieve', 'chitchat', or 'off-topic'")
    answer: str = Field(description="Grounded generated answer or conversational response")
    sources: list[Source] = Field(default_factory=list, description="List of cited source chunks")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="ok")
    vector_store_loaded: bool = Field(default=False)
    embedding_model: str = Field(default="")
