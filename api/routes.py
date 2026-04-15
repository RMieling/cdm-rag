from fastapi import APIRouter, Depends, HTTPException, Request

from api.config import AppConfig, ChatRequest, ChatResponse, get_config
from api.utils.logger import api_logger
from api.utils.utils import check_ollama_models_ready

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, payload: ChatRequest):
    """
    Receives a question, passes it to the LangGraph RAG pipeline,
    and returns the contextual answer.
    """
    api_logger.info(f"Received chat request for session: {payload.session_id}")

    # Extract our pre-loaded pipeline from the app state
    rag_pipeline = request.app.state.rag_pipeline

    if not rag_pipeline:
        api_logger.error("RAG Pipeline is not initialized in app state.")
        raise HTTPException(status_code=500, detail="AI Services are currently unavailable.")

    try:
        # Ask the pipeline
        result = rag_pipeline.ask(question=payload.question, session_id=payload.session_id)

        return ChatResponse(
            answer=result.get("answer", "I could not generate an answer."),
            sources=result.get("sources", []),
        )

    except Exception as e:
        api_logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing the query.") from e


@router.get("/health")
async def health_check():
    """Simple endpoint for Docker and Cloud load balancers to check if the app is alive."""
    return {"status": "healthy", "service": "cdm-rag-api"}


@router.get("/health/ai")
async def check_ai_health(config: AppConfig = Depends(get_config)):
    """
    Checks if the required LLM models are fully downloaded and ready.
    """
    if config.LLM_PROVIDER != "ollama":
        return {
            "status": "ready",
            "message": f"Using cloud provider: {config.LLM_PROVIDER}",
        }

    required_models = [config.OLLAMA_LLM_MODEL, config.OLLAMA_EMBED_MODEL]

    try:
        # NEW: Call our shared utility function
        is_ready = check_ollama_models_ready(config.OLLAMA_ENDPOINT, required_models)

        if is_ready:
            return {"status": "ready"}
        else:
            return {"status": "downloading"}

    except Exception as e:
        api_logger.warning(f"Ollama health check failed: {e}")
        return {"status": "unavailable"}
