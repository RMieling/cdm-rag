# CDM RAG - Common Data Model Question Answering

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

A **Retrieval-Augmented Generation (RAG)** system that answers natural language questions about the Microsoft Common Data Model (CDM) using local LLMs or cloud APIs. Features both a REST API and interactive web interface.

RAG Pipeline in development only, not production ready.

## Overview

CDM RAG enables users to query CDM entity definitions, attributes, and relationships conversationally. Instead of manually navigating schema documents, ask questions like:
- "What are the core attributes of the 'Account' entity?"
- "How does a 'Contact' relate to an 'Organization'?"
- "What entities exist for banking operations?"

The system retrieves relevant CDM schema documents and generates accurate, contextual answers powered by your choice of LLM.

The answers are generated based on the following RAG flow:
1. **Contextualize**: Contextualizes query based on chat history into standalone retrieval question
2. **Retrieve**: Generates a Cypher with which to query the Neo4j GraphDatabase (tries to self-correct thrice in case of occurring errors)
3. **Generate**: Generates a chat response based on the retrieved cypher result and displays the generated cypher as a source


## Key Features

- **Multi-LLM Support**: Use local Ollama (privacy-first) or OpenAI (faster)
- **Web UI + REST API**: Streamlit frontend for users, FastAPI backend for integrations
- **RAG Pipeline**: LangGraph-based retrieval + generation workflow
- **Docker Ready**: Docker Compose setup with Ollama, Neo4j, API, and frontend
- **Local-First**: Run entirely offline with Ollama + local embeddings (Optional)


## Project Structure

```
cdm-rag/
├── api/                       # FastAPI backend
│   ├── services/              # Business logic
│   │   ├── rag_pipeline.py    # LangGraph RAG workflow
│   │   ├── vector_store.py    # Neo4j GraphDatabase
│   │   └── parse_cdm.py       # CDM schema parsing
│   ├── routes.py              # REST API endpoints
│   ├── config.py              # Configuration management
│   ├── main.py                # FastAPI app entry
│   └── requirements.txt       # Backend dependencies
├── frontend/                  # Streamlit web UI
│   ├── app.py                 # Chat interface
│   └── requirements.txt       # Frontend dependencies
├── config/                    # Configuration files
│   ├── base.yaml              # RAG parameters
│   └── prompts.yaml           # LLM system prompts
├── data/
│   ├── CDM/                   # Common Data Model schema documents
│   └── output_schemas/        # Cached CDM schema documents
├── tests/                     # Test suite
├── docker-compose.yml         # Multi-container orchestration
├── credentials.yaml           # User login credentials
├── .env.example               # Environment template
└── pyproject.toml             # Project metadata & tool config
```

## Quick Start

### Option 1: Docker (Recommended)

**Requirements:** Docker & Docker Compose

```bash
# Clone and setup
git clone https://github.com/RMieling/cdm-rag.git
cd cdm-rag

# Configure environment and credentials
cp .env.example .env
cp credentials.yaml.example credentials.yaml
# Edit .env and .yaml as needed (see Configuration below)

# Start all services
docker-compose up -d --build

# Access services
# Frontend: http://localhost:8501
# API: http://localhost:8000
```

**Default Frontend Login:**
- Username: `testuser`
- Password: `testpassword`

### Option 2: Local Development

**Requirements:** Python 3.12+, Ollama (optional), or OpenAI API key

```bash
# Clone and setup
git clone https://github.com/RMieling/cdm-rag.git
cd cdm-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Terminal 1: Start backend API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start frontend
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

## Configuration

### Environment Variables (.env)

**LLM Provider Selection:**

```bash
# Use local Ollama (recommended for privacy)
LLM_PROVIDER=ollama
OLLAMA_ENDPOINT=http://ollama:11434
OLLAMA_LLM_MODEL=mistral              # LLM for answer generation
OLLAMA_EMBED_MODEL=nomic-embed-text   # Model for embeddings

# OR use OpenAI (faster)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-small
```

**Neo4j Graph Database:**
```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j_password
```

**RAG & Logging:**
```bash
ENVIRONMENT=DEV                  # DEV or PROD
LOG_LEVEL=DEBUG                  # DEBUG, INFO, WARNING, ERROR
ENABLE_FILE_LOGGING=true        # Write logs to logs/
```

### User Credentials

Edit `credentials.yaml` to manage frontend login accounts:

```yaml
credentials:
  usernames:
    jsmith:
      email: john@example.com
      name: John Smith
      password: $2b$12$hashed-password  # Use bcrypt hashing
```

To generate a bcrypt-hashed password:
```python
import streamlit_authenticator as stauth
hashed = stauth.Hasher.hash("your-password")
print(hashed)
```

## Usage

### Web Interface

1. **Login** with credentials from `credentials.yaml`
2. **View CDM Entities** in the sidebar
3. **Ask Questions** in the chat area
4. **Get Answers** with source citations from CDM schema documents

### REST API

**Ask a Question:**
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What attributes does the Account entity have?",
    "session_id": "user-123"
  }'
```

**Response:**
```json
{
  "answer": "The Account entity includes attributes such as...",
  "sources": [
    {
      "content": "Account entity definition...",
      "source": "Executed Cypher:"
    }
  ],
  "message_id": "msg-456"
}
```


## Development

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=api tests/

# Specific test file
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Format code
black api/ frontend/
isort api/ frontend/

# Lint and check
ruff check api/ frontend/

# Security audit
bandit -r api/

# Pre-commit hooks
pre-commit run --all-files
```

### Adding Features

1. Backend API: Edit `api/services/` and `api/routes.py`
2. Frontend: Edit `frontend/app.py`
3. Dependencies: Update `api/requirements.txt` or `frontend/requirements.txt`
4. Tests: Add to `tests/` and run `pytest`

## Troubleshooting

**Models not loading (Ollama provider):**
```bash
docker-compose logs ollama
docker exec ollama ollama pull mistral
docker exec ollama ollama pull nomic-embed-text
```

**API returns 503 Service Unavailable:**
- Wait 30-60 seconds for model initialization
- Check logs: `docker-compose logs app`
- Verify .env configuration

**Frontend won't connect:**
- Verify API is running on port 8000
- Check API_URL in environment: `echo $API_URL`
- View logs: `docker-compose logs frontend`

## Resources

- [Microsoft CDM](https://github.com/microsoft/CDM)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Ollama Models](https://ollama.ai)
- [OpenAI API Reference](https://platform.openai.com/docs)

---

**Ready to start?** Run `docker-compose up -d --build` and open http://localhost:8501!
