# Enterprise AI Knowledge Assistant

Intelligent document Q&A system with RAG, authentication, evaluation, and CI/CD.

**Current milestone: Task 1** — Python environment, LangChain chain, OpenAI integration, multi-turn chatbot.

## Tech stack

| Layer | Technology |
|-------|------------|
| RAG orchestration | LangChain |
| Vector store | ChromaDB |
| LLM | OpenAI / Gemini |
| API | FastAPI + JWT |
| UI | Streamlit |
| Ops | Docker, GitHub Actions, MLflow, RAGAS |
| Persistence | SQLite |

## Project structure

```
AI Knowledge Assistant/
├── app/
│   ├── config.py           # Environment settings
│   ├── chains/
│   │   └── qa_chain.py     # PromptTemplate + LLMChain + ChatOpenAI
│   └── chatbot/
│       └── multi_turn.py   # Multi-turn ChatModels conversation
├── scripts/
│   ├── test_openai.py      # First OpenAI API smoke test
│   ├── run_chain.py        # LangChain chain demo
│   └── chat_cli.py         # Interactive multi-turn CLI
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

### 1. Virtual environment

Dependencies are declared in `pyproject.toml` (PEP 621). Install the project in editable mode:

```powershell
# Windows (PowerShell) — project root
python -m venv gaproj
.\gaproj\Scripts\Activate.ps1
pip install -e ".[dev]"   # app + dev tools (pytest, python-docx)
# pip install -e .        # runtime only (no dev extras)
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
# Edit .env and set OPENAI_API_KEY=sk-...
```

Never commit `.env` — it is listed in `.gitignore`.

### 3. Verify installation

```powershell
# Direct OpenAI API call
python scripts/test_openai.py

# LangChain LLMChain (PromptTemplate + ChatOpenAI)
python scripts/run_chain.py

# Optional custom question
python scripts/run_chain.py "What is a vector database?"

# Multi-turn interactive chat
python scripts/chat_cli.py
```

### 4. Run tests (no API key required)

```powershell
pytest tests/ -v
```