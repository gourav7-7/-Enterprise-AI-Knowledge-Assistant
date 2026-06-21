# Enterprise AI Knowledge Assistant

A production-grade document Q&A system built with Retrieval-Augmented Generation (RAG). Upload PDFs, ask questions in natural language, and get answers grounded in your documents — with source citations, JWT-secured access, per-user chat history, and automated answer-quality evaluation.

> **Status:** Authenticated FastAPI backend is feature-complete and runs end-to-end on a local database. Remaining work: wiring the Streamlit UI to the API, Docker containerisation, and CI/CD.

## What it does

- **RAG pipeline** — PDF → text → chunked → embedded → stored in ChromaDB; questions are answered using only the retrieved context, with citations (source file + page).
- **Conversational memory** — follow-up questions are understood in context via a history-aware retriever.
- **Authentication** — register / login / logout with JWT; passwords hashed with Argon2. Protected endpoints reject unauthenticated requests.
- **Persistence** — users, chat history, and feedback stored via SQLAlchemy (SQLite locally; PostgreSQL-ready through a single connection-string change).
- **Evaluation** — RAGAS scores answer **faithfulness** and **relevancy** over a fixed test set.
- **Robustness** — structured logging, typed exception handling with clean JSON errors, retry/backoff on LLM calls, and an async query path.

## Architecture

```
Client (Streamlit UI / Swagger / any HTTP client)
        │  HTTP + JWT
        ▼
FastAPI backend  ── auth · upload · query · history · feedback
        │
        ├── RAG engine (LangChain): ingestion → retriever → chain
        │        ├── ChromaDB        (document vectors)
        │        └── OpenAI API      (embeddings + LLM)
        │
        └── SQLAlchemy → SQLite / PostgreSQL  (users, chat_history, feedback)
```


## Tech stack

| Layer | Technology |
|-------|------------|
| RAG orchestration | LangChain (LCEL, `create_retrieval_chain`) |
| Vector store | ChromaDB |
| Embeddings + LLM | OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) |
| API | FastAPI (async), Pydantic |
| Auth | PyJWT (JWT), pwdlib (Argon2) |
| Persistence | SQLAlchemy → SQLite (local) / PostgreSQL (production) |
| Evaluation | RAGAS (faithfulness, answer relevancy) |
| UI | Streamlit |
| Planned ops | Docker, docker-compose, GitHub Actions |

## Project structure

```
AI Knowledge Assistant/
├── app/
│   ├── main.py                 # FastAPI app — routers, exception handlers, /health
│   ├── config.py               # Settings loaded from environment
│   ├── dependencies.py         # DI providers: chain, ingestor, get_current_user
│   ├── api/
│   │   ├── auth.py             # /register, /login, /logout, /me
│   │   ├── documents.py        # /upload (PDF ingestion)
│   │   ├── query.py            # /query (protected, async, saves history)
│   │   ├── history.py          # /history (per-user past turns)
│   │   └── feedback.py         # /feedback
│   ├── core/
│   │   ├── logger.py           # Structured logging (text or JSON)
│   │   ├── exception.py        # Typed errors + FastAPI handlers
│   │   └── security.py         # Argon2 hashing + JWT create/decode
│   ├── db/
│   │   ├── database.py         # Engine, session, init_db, get_db
│   │   ├── models.py           # users, sessions, chat_history, feedback
│   │   └── crud.py             # DB access helpers
│   ├── rag/
│   │   ├── ingestion.py        # Load → split → embed → store (callable module)
│   │   ├── retriever.py        # Query → similarity search
│   │   ├── chain.py            # RAGChain + ConversationalRAGChain
│   │   └── evaluation.py       # RAGAS evaluation
│   ├── schemas/                # Pydantic request/response models
│   ├── chains/qa_chain.py      # Early LangChain LLMChain (foundations)
│   └── chatbot/multi_turn.py   # Multi-turn chatbot (foundations)
├── frontend/
│   └── streamlit_app.py        # UI (standalone; API wiring in progress)
├── scripts/
│   ├── test_openai.py          # OpenAI API smoke test
│   ├── run_chain.py            # LangChain chain demo
│   ├── chat_cli.py             # Multi-turn chat CLI
│   ├── ask_pdf.py              # Standalone PDF Q&A with citations
│   └── evaluate.py             # Run RAGAS over the eval set
├── tests/                      # pytest (config, qa_chain, ingestion)
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

### 1. Virtual environment

Dependencies are declared in `pyproject.toml` (PEP 621). Install in editable mode:

```powershell
# Windows (PowerShell) — project root
python -m venv gaproj
.\gaproj\Scripts\Activate.ps1
pip install -e ".[dev]"          # backend + dev tools (pytest)          
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Environment variables

```powershell
copy .env.example .env
```

Set the following in `.env`:

```dotenv
OPENAI_API_KEY="sk-..."                 # required
JWT_SECRET_KEY="<32-byte hex secret>"   # required — generate below
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
# DATABASE_URL is optional. Omit it to use SQLite (data/app.db).
# For PostgreSQL:
# DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/knowledge_assistant"
```

Generate a JWT secret:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

`.env` is gitignored — never commit it.

### 3. Run the API

```powershell

uvicorn app.main:app --reload --port 8000
```

Open the interactive docs at **http://127.0.0.1:8000/docs**. On first start the database tables are created automatically.

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET  | `/health`        | —   | Liveness check |
| POST | `/auth/register` | —   | Create a user |
| POST | `/auth/login`    | —   | Obtain a JWT (OAuth2 form) |
| POST | `/auth/logout`   | ✓   | Stateless logout (client discards token) |
| GET  | `/auth/me`       | ✓   | Current authenticated user |
| POST | `/upload`        | ✓   | Upload and ingest a PDF |
| POST | `/query`         | ✓   | Ask a question → grounded answer + sources |
| GET  | `/history`       | ✓   | Your past Q/A turns |
| POST | `/feedback`      | ✓   | Rate an answer |

**Quick end-to-end test in `/docs`:** register → click **Authorize** (login) → `/upload` a PDF → `/query` → `/history` → `/feedback`.

## Streamlit UI

```powershell
streamlit run frontend/streamlit_app.py
```

> The UI currently runs as a standalone demo that calls the RAG engine in-process (no login). Rewiring it to consume the authenticated FastAPI backend over HTTP is in progress.

## Evaluation (RAGAS)

Populate `data/eval/qa_pairs.json` with question / ground-truth pairs about your documents, then:

```powershell
python scripts/evaluate.py
```

This runs the real retriever + chain on each question and reports **faithfulness** and **answer relevancy**. It calls the OpenAI API, so it consumes tokens.

## Standalone scripts (no server required)

```powershell
python scripts/test_openai.py             # verify OpenAI connectivity
python scripts/run_chain.py "..."         # LangChain chain demo
python scripts/chat_cli.py                # multi-turn chat CLI
python scripts/ask_pdf.py data/uploads/your.pdf   # PDF Q&A with citations
```

## Tests

```powershell
pytest -v                # no API key required for the included tests
```

## Roadmap

- [x] LangChain foundations: chain, multi-turn chatbot
- [x] RAG pipeline: PDF ingestion, ChromaDB, retrieval with citations
- [x] Conversational RAG (history-aware retriever) + Streamlit demo
- [x] Modular, callable ingestion module + unit tests
- [x] FastAPI backend: upload, query, health, feedback
- [x] JWT authentication; token-protected endpoints
- [x] Persistence: users, chat history, feedback (SQLite; PostgreSQL-ready)
- [x] RAGAS evaluation (faithfulness, answer relevancy)
- [ ] Wire the Streamlit UI to the authenticated API
- [ ] Docker + docker-compose (with PostgreSQL service)
- [ ] CI/CD (GitHub Actions) and deployment
=======
pytest tests/ -v
```
