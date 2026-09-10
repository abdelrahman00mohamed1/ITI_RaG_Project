import os
import sys
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from api_client import RAGApiClient

load_dotenv()

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
client = RAGApiClient(base_url=DEFAULT_API_URL)

# Lazy-loaded in-process pipeline fallback if FastAPI backend is not running
_in_process_services = None


def _get_in_process_pipeline():
    global _in_process_services
    if _in_process_services is not None:
        return _in_process_services

    # Add backend directory to sys.path
    project_root = Path(__file__).resolve().parent.parent
    backend_path = project_root / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    try:
        from app.services.retrieval import ChromaRetrievalService
        from app.services.generation import GenerationService
        from app.core.config import settings

        # Find vector store directory (check backend/data or data/)
        possible_dirs = [
            project_root / "backend" / "data" / "vector_store",
            project_root / "data" / "vector_store",
            Path(settings.CHROMA_PERSIST_DIR),
        ]
        chosen_dir = None
        for p in possible_dirs:
            if p.exists() and (p / "chroma.sqlite3").exists():
                chosen_dir = str(p)
                break
        if not chosen_dir:
            chosen_dir = str(possible_dirs[0])

        retrieval = ChromaRetrievalService(persist_dir=chosen_dir)
        retrieval.initialize()

        generation = GenerationService()
        generation.initialize()

        _in_process_services = (retrieval, generation)
        return _in_process_services
    except Exception as e:
        print(f"Warning: could not initialize in-process RAG fallback: {e}")
        return None


def predict(message, history=None):
    if not message or not str(message).strip():
        return "Please ask a question about the Harry Potter books."

    query_str = str(message).strip()

    # 1. Try FastAPI backend first
    try:
        data = client.query(query_str)
        answer = data.get("answer", "No answer returned.")
        route = data.get("route", "retrieve")
        sources = data.get("sources", [])
    except Exception as api_err:
        # 2. Fallback to direct in-process RAG execution
        services = _get_in_process_pipeline()
        if services is not None:
            retrieval, generation = services
            route = generation.route_query(query_str)
            if route == "chitchat":
                answer = generation.handle_chitchat(query_str)
                sources = []
            elif route == "off-topic":
                answer = generation.handle_off_topic(query_str)
                sources = []
            else:
                context, sources_objs = retrieval.retrieve(query_str, top_k=3)
                answer = generation.generate_rag_answer(query_str, context)
                sources = [s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in sources_objs]
        else:
            return f"⚠️ Backend connection failed and in-process fallback unavailable: {api_err}"

    # Format output with route badge and citations
    output_parts = [f"**Route:** `{route.upper()}`\n\n{answer}"]

    if sources:
        output_parts.append("\n\n---\n### 📚 Cited Sources:")
        for s in sources:
            book = s.get("book_name", "Harry Potter")
            page = s.get("page_number", "N/A")
            score = s.get("score", 0.0)
            snippet = s.get("content_snippet", "")
            output_parts.append(
                f"\n* 📖 **{book}** (Page {page}) — *Relevance: {score:.2f}*\n  > _{snippet}_"
            )

    return "".join(output_parts)


def check_api_health(url):
    test_client = RAGApiClient(base_url=url)
    health = test_client.check_health()
    if health.get("status") == "ok":
        loaded = "✅ Loaded" if health.get("vector_store_loaded") else "⚠️ Empty"
        return f"Connected to {url} (Vector Store: {loaded})"
    
    # Check if in-process pipeline is ready
    services = _get_in_process_pipeline()
    if services is not None:
        retrieval, _ = services
        count = retrieval.collection.count() if retrieval.collection else 0
        return f"ℹ️ FastAPI offline. Running via direct in-process pipeline (ChromaDB records: {count})"

    return f"❌ Could not connect: {health.get('error', 'FastAPI offline')}"


custom_css = """
<style>
.gradio-container {
    background: radial-gradient(circle at 15% 10%, rgba(68, 44, 110, 0.4), transparent 30%),
                radial-gradient(circle at 85% 15%, rgba(26, 58, 88, 0.35), transparent 30%),
                #0e0b16 !important;
    color: #f5eedc !important;
}
h1, h2, h3 {
    color: #e5c276 !important;
}
</style>
"""

with gr.Blocks(title="Hogwarts Library Assistant") as demo:
    gr.HTML(custom_css)
    gr.Markdown(
        """
        # ⚡ Hogwarts Library Assistant
        ### Grounded Question-Answering over all 7 Harry Potter Books
        *Ask questions about characters, events, spells, or lore. Answers are strictly grounded in retrieved book pages.*
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=predict,
                examples=[
                    "Who is the Half-Blood Prince?",
                    "What is a Horcrux and how is it made?",
                    "What loophole did Mr Weasley write into the law about enchanting a car?",
                    "Good morning, how are you?",
                    "What is quantum mechanics?",
                ],
            )
        with gr.Column(scale=1):
            gr.Markdown("### 🏰 Connection Settings")
            api_url_input = gr.Textbox(label="FastAPI Base URL", value=DEFAULT_API_URL)
            health_btn = gr.Button("⚡ Test Connection")
            health_status = gr.Textbox(label="Status", interactive=False)
            health_btn.click(fn=check_api_health, inputs=api_url_input, outputs=health_status)

            gr.Markdown("---")
            gr.Markdown(
                """
                ### 📜 System Information
                - **Vector Store**: Local ChromaDB
                - **Embeddings**: `intfloat/multilingual-e5-large`
                - **Router**: Groq Llama 3
                - **RAG Generator**: Google Gemini
                """
            )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)

