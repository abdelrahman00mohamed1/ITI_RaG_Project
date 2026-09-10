import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Locate the root or backend directory for .env lookup
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # API Configuration
    PROJECT_NAME: str = "Harry Potter RAG Document Assistant"
    API_V1_STR: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    # Embedding Model
    EMBEDDING_MODEL: str = Field(
        default="intfloat/multilingual-e5-large",
        description="SentenceTransformer embedding model name",
    )

    # Local ChromaDB Vector Store
    CHROMA_PERSIST_DIR: str = Field(
        default="data/vector_store",
        description="Path to local persistent directory for ChromaDB",
    )
    CHROMA_COLLECTION: str = Field(
        default="harry_potter_books",
        description="ChromaDB collection name",
    )
    TOP_K: int = Field(
        default=3,
        description="Number of chunks to retrieve",
    )

    # Google Gemini LLM
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Generative AI API Key",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash-lite",
        description="Google Gemini model identifier",
    )

    # Groq API (Router & Chitchat)
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API Key",
    )
    GROQ_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model identifier for query routing",
    )

    model_config = SettingsConfigDict(
        env_file=(str(BASE_DIR / ".env"), str(BASE_DIR.parent / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
