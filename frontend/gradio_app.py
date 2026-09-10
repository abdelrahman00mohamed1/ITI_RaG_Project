import os
import gradio as gr
from dotenv import load_dotenv
from api_client import RAGApiClient

load_dotenv()

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
client = RAGApiClient(base_url=DEFAULT_API_URL)


def predict(message, history):
    if not message or not message.strip():
        return "Please ask a question about the Harry Potter books."

    try:
        data = client.query(message)
        answer = data.get("answer", "No answer returned.")
        route = data.get("route", "retrieve")
        sources = data.get("sources", [])

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
    except Exception as e:
        return f"⚠️ An error occurred: {e}"


def check_api_health(url):
    test_client = RAGApiClient(base_url=url)
    health = test_client.check_health()
    if health.get("status") == "ok":
        loaded = "✅ Loaded" if health.get("vector_store_loaded") else "⚠️ Empty"
        return f"Connected to {url} (Vector Store: {loaded})"
    return f"❌ Could not connect: {health.get('error', 'Unknown error')}"


custom_css = """
.gradio-container {
    background: radial-gradient(circle at 15% 10%, rgba(68, 44, 110, 0.4), transparent 30%),
                radial-gradient(circle at 85% 15%, rgba(26, 58, 88, 0.35), transparent 30%),
                #0e0b16 !important;
    color: #f5eedc !important;
}
h1, h2, h3 {
    color: #e5c276 !important;
}
"""

with gr.Blocks(css=custom_css, title="Hogwarts Library Assistant") as demo:
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
                type="messages",
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
    # share=True creates a free, secure public link in Google Colab automatically
    demo.launch(share=True, server_port=7860)
