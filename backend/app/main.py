from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import query
from app.core.config import settings
from app.services.generation import GenerationService
from app.services.retrieval import ChromaRetrievalService
from app.utils.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.
    Loads the embedding model and vector store ONCE during startup.
    """
    logger.info("Initializing Harry Potter RAG Application Services...")

    retrieval_service = ChromaRetrievalService()
    try:
        retrieval_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize Chroma retrieval service: {e}")

    generation_service = GenerationService()
    try:
        generation_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize generation service: {e}")

    # Attach to application state for access in route handlers
    app.state.retrieval_service = retrieval_service
    app.state.generation_service = generation_service

    logger.info("RAG Application Services initialization complete.")
    yield
    logger.info("Shutting down Harry Potter RAG Application Services...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade RAG system for searching and chatting with the Harry Potter book collection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(query.router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root_info():
    """Root endpoint providing project overview and documentation links."""
    return {
        "project": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "query_url": "/query",
        "status": "active",
    }


# Optional: Mount frontend static directory if present
web_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "web"
if web_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")
