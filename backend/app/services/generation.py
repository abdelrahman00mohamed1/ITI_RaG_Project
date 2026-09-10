import os
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.core.config import settings
from app.utils.logging_config import logger


ROUTER_SYSTEM_PROMPT = """You are an intent classification system for a Harry Potter Book Assistant.
Analyze the user message and classify it into EXACTLY ONE category.

Categories:
- retrieve: Questions about Harry Potter lore, characters, spells, plot, events, Hogwarts, artifacts, or book details.
- chitchat: Greetings (hello, hi), pleasantries, thanks, compliments, or casual conversational banter.
- off-topic: Inquiries about non-Harry Potter topics (e.g. math, coding, politics, weather, cooking, other movies).

Return ONLY the single label: "retrieve", "chitchat", or "off-topic". Do not add any punctuation or explanation."""

CHITCHAT_SYSTEM_PROMPT = """You are a warm, witty assistant in the Hogwarts Library.
Respond briefly and warmly to the user's greeting or comment with subtle Harry Potter charm.
Keep your response concise (1-2 sentences)."""

RAG_SYSTEM_PROMPT = """You are a strictly grounded AI assistant answering questions about the Harry Potter book collection.

STRICT GROUNDING INSTRUCTIONS:
1. Base your answer ONLY on the provided Context excerpts below.
2. Do NOT use any pre-trained external knowledge, facts, or assumptions outside of the provided Context.
3. If the provided Context does NOT contain enough factual information to answer the question accurately, you MUST reply with this exact sentence:
   "I do not know based on the provided context."
4. If the Context contains the answer, be concise, clear, and factual. You may cite the relevant book name and page number mentioned in the context brackets."""


def _extract_text(content) -> str:
    """Safely extracts text whether content is a string, list of dicts, or objects."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


class GenerationService:
    """Handles query routing, conversational chit-chat, and grounded Gemini RAG generation."""

    def __init__(self):
        self.gemini_llm: Optional[ChatGoogleGenerativeAI] = None
        self.groq_llm: Optional[ChatGroq] = None

    def initialize(self):
        """Initializes LLM clients."""
        # Initialize Gemini for RAG Generation
        if settings.GEMINI_API_KEY:
            try:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.0,
                )
                logger.info(f"Gemini LLM initialized with model: {settings.GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini LLM: {e}")
        else:
            logger.warning("GEMINI_API_KEY is not set. RAG generation will require mocked or direct response.")

        # Initialize Groq for Router & Chitchat
        if settings.GROQ_API_KEY:
            try:
                self.groq_llm = ChatGroq(
                    model=settings.GROQ_MODEL,
                    api_key=settings.GROQ_API_KEY,
                    temperature=0.0,
                )
                logger.info(f"Groq LLM initialized with model: {settings.GROQ_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq LLM: {e}")
        else:
            logger.warning("GROQ_API_KEY is not set. Heuristic router will be used as fallback.")

    def route_query(self, query: str) -> str:
        """Determines whether a query needs retrieval, chitchat, or is off-topic."""
        # Use Groq if available
        if self.groq_llm:
            try:
                response = self.groq_llm.invoke([
                    SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ])
                route = _extract_text(response.content).lower()
                for valid_route in ("retrieve", "chitchat", "off-topic"):
                    if valid_route in route:
                        return valid_route
            except Exception as e:
                logger.error(f"Groq routing error: {e}. Falling back to rule-based classification.")

        # Heuristic fallback if LLM is unreachable or key missing
        q_cleaned = query.strip(" '\"`.,!?;:").lower()
        greetings = [
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
            "how are you", "howdy", "greetings", "thanks", "thank you", "bye", "goodbye"
        ]
        if any(q_cleaned == g or q_cleaned.startswith(g) or g in q_cleaned for g in greetings):
            return "chitchat"

        off_topics = [
            "weather", "math", "calculator", "crypto", "bitcoin", "football",
            "basketball", "recipe", "bake", "python code", "capital of", "quantum"
        ]
        if any(ot in q_cleaned for ot in off_topics):
            return "off-topic"

        # Default to retrieve for book queries
        return "retrieve"

    def handle_chitchat(self, query: str) -> str:
        """Returns friendly Hogwarts-themed greeting."""
        if self.groq_llm:
            try:
                response = self.groq_llm.invoke([
                    SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ])
                return _extract_text(response.content)
            except Exception as e:
                logger.error(f"Chitchat generation error: {e}")

        return "Welcome to the Hogwarts Library! Ask me any question about the Harry Potter books, and I will search the restricted section for you."

    def handle_off_topic(self, query: str) -> str:
        """Politely informs user that the system is restricted to Harry Potter lore."""
        return (
            "I am a specialized Harry Potter Document Assistant. I can only assist with "
            "questions related to the Harry Potter book collection. Please feel free to ask about "
            "Hogwarts, characters, spells, or events!"
        )

    def generate_rag_answer(self, query: str, context: str) -> str:
        """Generates grounded answer using retrieved context via Gemini LLM."""
        if not context or not context.strip():
            return "I do not know based on the provided context."

        if not self.gemini_llm:
            return "Gemini LLM is not configured. Please supply a valid GEMINI_API_KEY in your .env file."

        user_content = f"Context excerpts:\n{context}\n\nQuestion:\n{query}"

        try:
            response = self.gemini_llm.invoke([
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            return _extract_text(response.content)
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"An error occurred while communicating with the generation model: {e}"
