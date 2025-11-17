# JR AutoRAG

JR AutoRAG is a local-first Retrieval Augmented Generation (RAG) workbench. A FastAPI backend orchestrates ingestion, retrieval, and provider calls, while a Bun + React admin console provides a single pane of glass for:

- Auto-detecting Ollama/LM Studio runtimes and switching providers instantly
- Configuring planner/gatherer/generator models and retrieval defaults
- Drag-and-drop ingestion for PDF, DOC/DOCX, Markdown, TXT (with OCR fallback for scanned PDFs)
- Running queries/evaluations and inspecting traces
- Managing documents (metadata, deletion, upload timestamps)

## Architecture

```text
┌────────────────┐      ┌──────────────────┐
│ Bun + React UI │ ───▶ │ FastAPI Backend │ ──▶ Providers (Ollama / LM Studio / Cloud)
└────────────────┘      └──────────────────┘
         │                          │
         └────── Documents / Traces ─┘ (JSON stores under `data/`)
```

## Prerequisites

- **Bun** v1.3+
- **Python** 3.11 with `pip`
- **OCR tooling** (for scanned PDFs):
  - macOS: `brew install tesseract poppler`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr poppler-utils`
  - Fedora: `sudo dnf install tesseract poppler-utils`
- Optional: Docker + Docker Compose

## Installation

```bash
# frontend & shared tooling
bun install

# backend deps (includes pdf2image / pytesseract / docx2txt)
cd api && python3 -m pip install -r requirements.txt
```

## Running locally

```bash
# backend (from api/)
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# frontend (from repo root)
bun dev        # http://localhost:3000
```

The frontend points to `http://localhost:8000` by default. Override with `BUN_PUBLIC_API_BASE_URL` (or `VITE_API_BASE_URL`) if your API lives elsewhere.

## Docker Compose

```bash
docker compose up --build
```

- API → <http://localhost:8000>
- Admin console → <http://localhost:3000>

## Document ingestion guide

| Input type | How it’s handled |
|------------|------------------|
| PDF        | Native text extraction via `pypdf`, then OCR fallback using `pdf2image` + `pytesseract` if the file is scanned |
| DOC/DOCX   | Processed with `docx2txt` via a temporary file |
| Markdown   | `.md` / `.markdown` decoded to UTF-8 with light token stripping |
| TXT        | UTF-8 decode with `errors="ignore"` |
| Inline text | Use the “Ingest Text” form for quick snippets |

The admin console surfaces upload metadata (filename, content type, file size, timestamp) and allows deleting a document, which triggers a retrieval index rebuild automatically.

### Tips

1. Start Ollama (`ollama serve`) or LM Studio before launching the UI so the Quick Setup card can auto-detect local providers.
2. Large PDFs with OCR may take a few seconds—watch the status banner for progress.
3. Use the profiles dropdown to save favorite provider configurations (local vs. cloud).

## Testing & quality checks

```bash
# backend integration tests
cd api && PYTHONPATH=. pytest

# frontend tests
bun test

# build for production (static assets in dist/)
bun run build
```

You can combine them into a single sequence when prepping for demos:

```bash
cd api && PYTHONPATH=. pytest && cd .. && bun run build && bun test
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| PDF uploads return empty text | Ensure `pdf2image` and `pytesseract` are installed and that the system Tesseract binary is on `PATH`. On macOS run `brew install tesseract poppler`. |
| Provider buttons don’t save | Confirm the FastAPI backend is running on `:8000` and that CORS isn’t blocked. Check the status badge in the UI header. |
| Ollama/LM Studio not detected | Click **Rescan**, ensure the runtime listens on the default ports (`11434` / `1234`), or edit `JR_OLLAMA_URL` / `JR_LMSTUDIO_URL`. |

## Deployment

1. Run `bun run build` and host the `dist/` directory behind any static host (Bun, nginx, S3 + CloudFront, etc.).
2. Deploy the FastAPI app (Uvicorn/Gunicorn, Fly.io, Render, etc.). Point the frontend’s `BUN_PUBLIC_API_BASE_URL` to the deployed API.
3. Persist the `data/` directory (config, documents, traces) on shared storage or a volume for stateful runs.

## Onboarding

An abridged onboarding checklist lives in [`onboarding.txt`](./onboarding.txt).
