import os
from pathlib import Path
from typing import Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.schemas.query import Source
from app.utils.logging_config import logger


class ChromaRetrievalService:
    """Service responsible for querying the local persistent ChromaDB vector store."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.collection = None

    def initialize(self):
        """Loads the embedding model and connects to local ChromaDB once at startup."""
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Resolve path relative to backend root if not absolute
        abs_persist_path = Path(self.persist_dir)
        if not abs_persist_path.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent.parent
            abs_persist_path = backend_root / self.persist_dir

        abs_persist_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Connecting to persistent ChromaDB at: {abs_persist_path}")

        self.chroma_client = chromadb.PersistentClient(path=str(abs_persist_path))
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            count = self.collection.count()
            logger.info(f"Chroma collection '{self.collection_name}' initialized with {count} records.")
        except Exception as e:
            logger.error(f"Error accessing collection '{self.collection_name}': {e}")
            self.collection = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[str, list[Source]]:
        """Encodes query with E5 prefix and retrieves top_k matching chunks from ChromaDB."""
        if not self.collection or self.collection.count() == 0:
            logger.warning("Vector store is empty or uninitialized. Run the notebook to populate it.")
            return "", []

        k = top_k or settings.TOP_K

        # E5 models require 'query: ' prefix for asymmetric retrieval queries
        formatted_query = f"query: {query.strip()}"
        query_vector = self.model.encode([formatted_query], normalize_embeddings=True)[0].tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Error executing ChromaDB query: {e}")
            return "", []

        sources: list[Source] = []
        context_parts: list[str] = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 0.0
            # Convert cosine distance to approximate similarity score (1 - distance)
            score = round(max(0.0, 1.0 - dist), 4)

            book_name = meta.get("book_name", "Harry Potter")
            page_number = int(meta.get("page_number", 0))
            chunk_id = str(meta.get("chunk_id", f"chunk_{i}"))

            sources.append(
                Source(
                    book_name=book_name,
                    page_number=page_number,
                    score=score,
                    chunk_id=chunk_id,
                    content_snippet=doc[:150] + "..." if len(doc) > 150 else doc,
                )
            )

            context_parts.append(
                f"[Book: {book_name} | Page: {page_number}]\n{doc}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, sources
