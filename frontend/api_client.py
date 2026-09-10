import os
from typing import Any, Dict
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class RAGApiClient:
    """Client for interacting with the Harry Potter FastAPI backend."""

    def __init__(self, base_url: str = DEFAULT_API_BASE_URL, timeout: int = 45):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Queries the /health endpoint to verify API connectivity."""
        url = f"{self.base_url}/health"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": str(e), "vector_store_loaded": False}

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Submits a query to the /query endpoint."""
        url = f"{self.base_url}/query"
        payload = {"query": question, "top_k": top_k}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"API Error ({response.status_code}): {detail}") from e
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Could not connect to FastAPI at '{self.base_url}'. Make sure the backend server is running."
            ) from e
