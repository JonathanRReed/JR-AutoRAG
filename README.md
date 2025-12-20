# JR AutoRAG

**JR AutoRAG** is a 100% local Retrieval Augmented Generation workbench. Run your own FastAPI backend and Bun/React admin console to ingest docs, configure local LLMs (Ollama / LM Studio), and inspect the full pipeline (planning → retrieval → generation) with transparent traces.

---

## At a glance

- **Local-first**: No cloud required; keep data on-device.
- **Provider auto-detect**: Ollama / LM Studio discovery with one-click config save.
- **Document ingestion**: Drag/drop PDFs, DOC/DOCX, Markdown, TXT + inline text; OCR fallback for scans.
- **Presets & profiles**: Fast / Balanced / Thorough retrieval presets; save provider profiles.
- **Observability**: Per-step traces, tokens, timing, and errors.

## Architecture

```text
┌────────────────┐      ┌──────────────────┐
│ Bun + React UI │ ───▶ │ FastAPI Backend │ ──▶ Providers (Ollama / LM Studio / Cloud)
└────────────────┘      └──────────────────┘
         │                          │
         └────── Documents / Traces ─┘ (JSON stores under `data/`)
```

## Quickstart (5 minutes)

```bash
# 1) Install deps
bun install
cd api && python3 -m pip install -r requirements.txt

# 2) Run everything (from repo root)
bun run dev:all
# UI: http://localhost:3000  | API: http://localhost:8000

# 3) In the UI header, click Test next to API Base URL — expect “API reachable”
```

## Requirements

- Bun v1.3+
- Python 3.11 + pip
- OCR (for scanned PDFs):
  - macOS: `brew install tesseract poppler`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr poppler-utils`
  - Fedora: `sudo dnf install tesseract poppler-utils`
- Optional: Docker + Docker Compose

## Local development (manual control)

```bash
# backend (from api/)
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# frontend (from repo root)
bun dev   # http://localhost:3000
```

Override API target with `BUN_PUBLIC_API_BASE_URL` (or `VITE_API_BASE_URL`).

## Docker Compose

```bash
docker compose up --build
```

- API → <http://localhost:8000>
- Admin console → <http://localhost:3000>

## Document ingestion behavior

| Input type | How it’s handled |
|------------|------------------|
| PDF        | `pypdf` for text; OCR fallback via `pdf2image` + `pytesseract` for scans |
| DOC/DOCX   | `docx2txt` via temp file |
| Markdown   | UTF-8 decode with light token stripping |
| TXT        | UTF-8 decode with `errors="ignore"` |
| Inline text | “Ingest Text” form for quick snippets |

The admin console shows upload metadata and allows deletion, which triggers an index rebuild.

## Retrieval presets

| Preset | Top-K | Target Tokens | Coverage | Best for |
|--------|-------|---------------|----------|----------|
| Fast | 3 | 800 | 50% | Low-latency answers |
| Balanced (default) | 5 | 1600 | 70% | General use |
| Thorough | 10 | 3000 | 90% | Deep research |

Apply via API:

```bash
curl -X POST http://localhost:8000/config/presets/balanced
```

## Observability (per-query trace)

1. Planning: generated search queries, target tokens, coverage goals
2. Retrieval: chunks per sub-query, timing, unique sources
3. Generation: provider, model, context tokens, errors

## Configuration & env

- `BUN_PUBLIC_API_BASE_URL` (or `VITE_API_BASE_URL`): frontend → API URL
- `JR_OLLAMA_URL`, `JR_LMSTUDIO_URL`: override local provider endpoints
- `JR_DATA_DIR` (backend): data/config/traces storage path

## Testing & quality checks

```bash
# backend integration tests
cd api && PYTHONPATH=. pytest

# frontend tests
bun test

# production build
bun run build
```

For demo prep:

```bash
cd api && PYTHONPATH=. pytest && cd .. && bun run build && bun test
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| PDF uploads return empty text | Ensure `pypdf`, `pdf2image`, `pytesseract` installed and `pdftotext`/`tesseract` on PATH. On macOS: `brew install poppler tesseract`, then restart API. |
| Provider buttons don’t save | Confirm FastAPI running on `:8000`; check UI status badge; ensure CORS not blocked. |
| Ollama/LM Studio not detected | Click **Rescan**; ensure runtimes listen on `11434` / `1234`; override `JR_OLLAMA_URL` / `JR_LMSTUDIO_URL`. |

## Deployment

1. `bun run build` and serve `dist/` (Bun, nginx, S3+CloudFront, etc.).
2. Deploy FastAPI (Uvicorn/Gunicorn, Fly.io, Render, etc.). Point frontend `BUN_PUBLIC_API_BASE_URL` to it.
3. Persist `data/` (config, documents, traces) on shared storage/volume for stateful runs.

## Onboarding

See the refreshed checklist in [`onboarding.txt`](./onboarding.txt) for a 5-minute path from clone → first answer.
