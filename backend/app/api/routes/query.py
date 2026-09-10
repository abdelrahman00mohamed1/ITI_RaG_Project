from fastapi import APIRouter, HTTPException, Request, status
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.utils.logging_config import logger

router = APIRouter(tags=["Query & Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Returns application status and indicates whether the vector store is loaded."""
    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    vector_loaded = False
    if retrieval_service and retrieval_service.collection is not None:
        try:
            vector_loaded = retrieval_service.collection.count() > 0
        except Exception:
            vector_loaded = False

    return HealthResponse(
        status="ok",
        vector_store_loaded=vector_loaded,
        embedding_model=getattr(retrieval_service, "model_name", "configured"),
    )


@router.post("/query", response_model=QueryResponse)
def process_query(request_body: QueryRequest, request: Request) -> QueryResponse:
    """Routes and answers user questions grounded in the Harry Potter document collection."""
    user_query = request_body.query or request_body.question
    if not user_query or not user_query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty",
        )

    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    generation_service = getattr(request.app.state, "generation_service", None)

    if not retrieval_service or not generation_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core RAG services are not properly initialized.",
        )

    # 1. Route the query (retrieve, chitchat, or off-topic)
    route = generation_service.route_query(user_query)
    logger.info(f"Query: '{user_query}' -> Route: '{route}'")

    # 2. Handle Chitchat
    if route == "chitchat":
        answer = generation_service.handle_chitchat(user_query)
        return QueryResponse(
            query=user_query,
            route=route,
            answer=answer,
            sources=[],
        )

    # 3. Handle Off-topic
    if route == "off-topic":
        answer = generation_service.handle_off_topic(user_query)
        return QueryResponse(
            query=user_query,
            route=route,
            answer=answer,
            sources=[],
        )

    # 4. Handle Retrieval and Grounded Generation
    context, sources = retrieval_service.retrieve(user_query, top_k=request_body.top_k)

    if not sources or not context.strip():
        return QueryResponse(
            query=user_query,
            route="retrieve",
            answer="I do not know based on the provided context.",
            sources=[],
        )

    answer = generation_service.generate_rag_answer(user_query, context)

    return QueryResponse(
        query=user_query,
        route="retrieve",
        answer=answer,
        sources=sources,
    )
