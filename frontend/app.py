import os
import streamlit as st
from api_client import RAGApiClient

# Page configuration
st.set_page_config(
    page_title="Harry Potter RAG Library Assistant",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS styling for a magical, polished Harry Potter theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(68, 44, 110, 0.4), transparent 30%),
                    radial-gradient(circle at 85% 15%, rgba(26, 58, 88, 0.35), transparent 30%),
                    #0e0b16;
        color: #f5eedc;
        font-family: 'DM Sans', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #e5c276 !important;
    }
    
    .crest-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 18px 24px;
        border-radius: 14px;
        background: rgba(30, 24, 44, 0.7);
        border: 1px solid rgba(229, 194, 118, 0.25);
        margin-bottom: 24px;
    }
    
    .crest-icon {
        font-size: 2.4rem;
        background: rgba(229, 194, 118, 0.15);
        border: 1px solid #e5c276;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #e5c276;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
    
    .route-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .badge-retrieve { background: rgba(115, 214, 160, 0.2); color: #73d6a0; border: 1px solid #73d6a0; }
    .badge-chitchat { background: rgba(110, 168, 254, 0.2); color: #6ea8fe; border: 1px solid #6ea8fe; }
    .badge-offtopic { background: rgba(255, 138, 138, 0.2); color: #ff8a8a; border: 1px solid #ff8a8a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🏰 Connection Settings")
    default_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    backend_url = st.text_input("FastAPI Base URL", value=default_url)
    client = RAGApiClient(base_url=backend_url)

    if st.button("⚡ Test Connection", use_container_width=True):
        health = client.check_health()
        if health.get("status") == "ok":
            v_loaded = "✅ Loaded" if health.get("vector_store_loaded") else "⚠️ Empty (Run Notebook)"
            st.success(f"Connected! Vector Store: {v_loaded}")
        else:
            st.error(f"Failed to connect: {health.get('error', 'Unknown error')}")

    st.markdown("---")
    st.markdown("### 📜 System Information")
    st.caption("• **Vector Store**: Local ChromaDB")
    st.caption("• **Embeddings**: `intfloat/multilingual-e5-large`")
    st.caption("• **Routing**: Groq Llama 3")
    st.caption("• **RAG Generator**: Google Gemini")
    
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header banner
st.markdown(
    """
    <div class="crest-header">
        <div class="crest-icon">⚡</div>
        <div>
            <h1 style="margin: 0; font-size: 1.8rem;">Hogwarts Library Assistant</h1>
            <p style="margin: 2px 0 0 0; color: #a99ec0; font-size: 0.9rem;">
                Grounded Question-Answering over all 7 Harry Potter Books
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Greetings, traveler! I am the Hogwarts Library Assistant. Ask me anything regarding characters, spells, or events from the books, and I shall consult the archives for grounded answers.",
            "sources": [],
            "route": "chitchat",
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            badge_class = f"badge-{msg['route'].replace('-', '')}"
            st.markdown(
                f'<span class="route-badge {badge_class}">{msg["route"]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        
        # Display sources if available
        if msg.get("sources"):
            with st.expander(f"📚 Cited Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>📖 {src['book_name']}</strong> &nbsp;|&nbsp; <strong>Page {src['page_number']}</strong> &nbsp;|&nbsp; <em>Relevance: {src.get('score', 0):.2f}</em>
                            <p style="margin-top: 5px; color: #d8cfc4; font-size: 0.82rem;">{src.get('content_snippet', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Example suggestions
example_cols = st.columns(3)
example_questions = [
    "Who is the Half-Blood Prince?",
    "What is a Horcrux and how is it made?",
    "What loophole did Mr Weasley write into the law about enchanting a car?",
]

selected_example = None
for i, col in enumerate(example_cols):
    if col.button(f"🔍 {example_questions[i]}", key=f"ex_{i}", use_container_width=True):
        selected_example = example_questions[i]

user_prompt = st.chat_input("Ask a question about the Harry Potter books...") or selected_example

if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Call Backend API
    with st.chat_message("assistant"):
        with st.spinner("Consulting the Hogwarts archives..."):
            try:
                response = client.query(user_prompt)
                answer = response.get("answer", "No answer returned.")
                sources = response.get("sources", [])
                route = response.get("route", "retrieve")

                badge_class = f"badge-{route.replace('-', '')}"
                st.markdown(
                    f'<span class="route-badge {badge_class}">{route}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Cited Sources ({len(sources)})"):
                        for src in sources:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <strong>📖 {src['book_name']}</strong> &nbsp;|&nbsp; <strong>Page {src['page_number']}</strong> &nbsp;|&nbsp; <em>Relevance: {src.get('score', 0):.2f}</em>
                                    <p style="margin-top: 5px; color: #d8cfc4; font-size: 0.82rem;">{src.get('content_snippet', '')}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "route": route,
                })
            except Exception as e:
                error_msg = f"⚠️ An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "route": "error",
                })
