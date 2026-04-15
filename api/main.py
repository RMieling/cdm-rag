import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_config
from api.routes import router as api_router
from api.services.parse_cdm import ensure_manifest_resolved
from api.services.rag_pipeline import RAGPipeline
from api.services.vector_store import Neo4jGraphManager  # Updated import name
from api.utils.logger import main_logger
from api.utils.utils import check_ollama_models_ready


async def wait_for_ollama_models(endpoint: str, required_models: list, max_retries=10):
    """Polls the Ollama API until the required models are fully downloaded and ready."""
    main_logger.info(f"Checking Ollama status at {endpoint}...")

    for attempt in range(max_retries):
        try:
            if check_ollama_models_ready(endpoint, required_models):
                main_logger.info("All Ollama models are loaded and ready!")
                return True
            else:
                main_logger.info(
                    f"Models {required_models} still downloading. Waiting... (Attempt {attempt + 1}/{max_retries})"
                )
        except Exception:
            main_logger.warning(f"Cannot reach Ollama server yet. Booting up... (Attempt {attempt + 1}/{max_retries})")

        await asyncio.sleep(5)

    raise RuntimeError("Timed out waiting for Ollama models to become available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Initialize DB connections, ingest initial data, and load LLMs into memory.
    """
    main_logger.info("Starting up CDM RAG API...")
    config = get_config()

    try:
        # Optional: Wait for Ollama to boot up if you are using it locally
        if getattr(config, "LLM_PROVIDER", "").lower() == "ollama":
            required_models = [config.OLLAMA_EMBED_MODEL, config.OLLAMA_LLM_MODEL]
            await wait_for_ollama_models(config.OLLAMA_ENDPOINT, required_models)

        # Initialize Graph Database Manager and attach to app state
        db_manager = Neo4jGraphManager(config=config)
        app.state.db_manager = db_manager

        # TODO load based on config
        # Define the targeted manifests to ingest, and set flag for recursive traversal
        traverse_manifest_files = {
            "/core/applicationCommon/applicationCommon.manifest.cdm.json": False,
            "/core/operationsCommon/Entities/Common/Common.manifest.cdm.json": True,
            "/core/operationsCommon/Entities/Finance/Finance.manifest.cdm.json": True,
            "/FinancialServices/FinancialServices.manifest.cdm.json": True,
        }

        main_logger.info("Checking if CDM schemas are resolved and cached locally...")
        await ensure_manifest_resolved(traverse_manifest_files)

        # Trigger Ingestion
        main_logger.info("Ingesting CDM graph schema in Neo4j...")
        await db_manager.ingest_manifests(traverse_manifest_files=traverse_manifest_files, load_cached_resolved=True)

        # Initialize RAG Pipeline and attach to app state
        app.state.rag_pipeline = RAGPipeline(config=config, db_manager=db_manager)
        main_logger.info("LLM and Neo4j vector store successfully loaded into memory. API is ready!")

        yield

    except Exception as e:
        main_logger.critical(f"Failed to initialize AI services: {e}", exc_info=True)
        raise e
    finally:
        main_logger.info("Shutting down API... Cleaning up connections.")
        if hasattr(app.state, "db_manager") and app.state.db_manager:
            app.state.db_manager.close()
            main_logger.info("Neo4j driver connection closed successfully.")


# Initialize the FastAPI application
app = FastAPI(
    title="cdm-rag",
    description="RAG Pipeline for answering questions with respect to Microsofts Common Data Model CDM",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach our API routes
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    # Test for local dev
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
