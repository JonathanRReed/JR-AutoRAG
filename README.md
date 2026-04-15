# JR AutoRAG

**JR AutoRAG** is a 100% local, enterprise-ready Retrieval Augmented Generation workbench. Run your own FastAPI backend and Bun/React admin console to ingest docs, configure local LLMs (Ollama / LM Studio), and inspect the full pipeline with transparent traces.

**v3.0** brings production-grade security, reproducibility features, evaluation gates, and a plugin architecture.

---

## At a glance

- **Local-first**: No cloud required; keep data on-device.
- **Production-ready security**: API key auth, rate limiting, secrets vault, and safe defaults.
- **Provider auto-detect**: Ollama / LM Studio discovery with one-click config save.
- **Document ingestion**: Drag/drop PDFs, DOC/DOCX, Markdown, TXT + inline text; OCR fallback for scans.
- **Presets & profiles**: Fast / Balanced / Thorough retrieval presets; save provider profiles.
- **Observability**: Per-step traces, tokens, timing, and errors.
- **Evaluation gates**: Built-in benchmarks with pass/fail thresholds for CI/CD.
- **Plugin architecture**: Extend with custom retrievers, chunkers, and more.

## What's New in v3.0

| Feature | Description |
|---------|-------------|
| **Security Middleware** | API key auth, rate limiting, request size limits, security headers |
| **Secrets Vault** | OS keychain integration + encrypted vault for API keys |
| **Config Snapshots** | Immutable configuration snapshots for reproducibility |
| **Trace Replay & Diff** | Replay past queries and compare traces |
| **Evaluation Gates** | Hard thresholds that fail builds if quality drops |
| **Loop Budgets** | Max iterations, tokens, and time limits for iterative loops |
| **Answerability Calibration** | Multi-dimensional scoring for abstention decisions |
| **Span-Level Citations** | Character-offset citations with claim extraction |
| **Document ACLs** | Per-document access control and query-time filtering |
| **PII Detection** | Detect and redact sensitive information |
| **Plugin Architecture** | Stable ABCs for custom components |

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

# 3) In the UI header, click Test next to API Base URL — expect "API reachable"
```

## Requirements

- Bun v1.3+
- Python 3.11 + pip
- OCR (for scanned PDFs):
  - macOS: `brew install tesseract poppler`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr poppler-utils`
  - Fedora: `sudo dnf install tesseract poppler-utils`
- Optional: Docker + Docker Compose

## Security

JR AutoRAG ships with **secure defaults**:

- **Localhost-only binding** by default (use `--expose` flag for network access)
- **API key authentication** (enable via `AUTORAG_AUTH_ENABLED=true`)
- **Rate limiting** (100 requests/minute default)
- **Request size limits** (50MB default)
- **Security headers** (XSS, clickjacking, content-type sniffing protection)

See [SECURITY.md](./Public/SECURITY.md) for production deployment guidance including:
- TLS configuration with Nginx/Caddy
- Secrets management best practices
- Reverse proxy recipes

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTORAG_AUTH_ENABLED` | `false` | Enable API key authentication |
| `AUTORAG_API_KEYS` | - | Comma-separated API keys |
| `AUTORAG_ALLOWED_ORIGINS` | `localhost` | CORS allowed origins |
| `AUTORAG_EXPOSE` | `false` | Allow non-localhost binding |
| `AUTORAG_RATE_LIMIT_ENABLED` | `true` | Enable request rate limiting |
| `AUTORAG_RATE_LIMIT_RPM` | `100` | Requests per minute limit |

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

| Input type | How it's handled |
|------------|------------------|
| PDF        | `pypdf` for text; OCR fallback via `pdf2image` + `pytesseract` for scans |
| DOC/DOCX   | `docx2txt` via temp file |
| Markdown   | UTF-8 decode with light token stripping |
| TXT        | UTF-8 decode with `errors="ignore"` |
| Inline text | "Ingest Text" form for quick snippets |

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

## Evaluation Gates

Run benchmarks with quality thresholds in CI:

```python
from app.core.eval_gates import GatedEvaluator, EvalThresholds

evaluator = GatedEvaluator(thresholds=EvalThresholds.strict())
result = await evaluator.evaluate_with_gates(orchestrator, "my_golden_set")

if not result.all_passed:
    print(f"Build FAILED: {result.failed_gates}")
    sys.exit(1)
```

## Plugin Architecture

Extend the system with custom components:

```python
from app.plugins import RetrieverPlugin, PluginInfo, PluginType

class MyRetriever(RetrieverPlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="my_retriever",
            plugin_type=PluginType.RETRIEVER,
            version="1.0.0",
            description="Custom retriever",
        )
    
    def retrieve(self, query: str, k: int = 10):
        # Your retrieval logic
        pass

def create_plugin():  # Required for auto-discovery
    return MyRetriever()
```

## Observability (per-query trace)

1. Planning: generated search queries, target tokens, coverage goals
2. Retrieval: chunks per sub-query, timing, unique sources
3. Generation: provider, model, context tokens, errors
4. Configuration snapshot: model, retrieval settings, corpus hash

## Testing & quality checks

```bash
# backend integration tests
cd api && PYTHONPATH=. pytest

# frontend tests
bun test

# production build
bun run build

# run evaluation gates
python -c "from app.core.eval_gates import run_eval_gates_cli; exit(run_eval_gates_cli('default'))"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| PDF uploads return empty text | Ensure `pypdf`, `pdf2image`, `pytesseract` installed and `pdftotext`/`tesseract` on PATH. On macOS: `brew install poppler tesseract`, then restart API. |
| Provider buttons don't save | Confirm FastAPI running on `:8000`; check UI status badge; ensure CORS not blocked. |
| Ollama/LM Studio not detected | Click **Rescan**; ensure runtimes listen on `11434` / `1234`; override `JR_OLLAMA_URL` / `JR_LMSTUDIO_URL`. |
| Authentication errors | Set `AUTORAG_AUTH_ENABLED=true` and `AUTORAG_API_KEYS=your-key`. |

## Deployment

1. `bun run build` and serve `dist/` (Bun, nginx, S3+CloudFront, etc.).
2. Deploy FastAPI (Uvicorn/Gunicorn, Fly.io, Render, etc.). Point frontend `BUN_PUBLIC_API_BASE_URL` to it.
3. **Enable security**: Set `AUTORAG_AUTH_ENABLED=true` and configure API keys.
4. **Use TLS**: Put behind Nginx/Caddy with HTTPS (see [SECURITY.md](./Public/SECURITY.md)).
5. Persist `data/` (config, documents, traces) on shared storage/volume for stateful runs.

## Onboarding

See the checklist in [`Public/onboarding.txt`](./Public/onboarding.txt) for a 5-minute path from clone → first answer.

## Documentation

- [SECURITY.md](./Public/SECURITY.md) - Security configuration and production deployment
- [Product.md](./Product.md) - Product overview and architecture details

## License

Licensed under the Functional Source License, Version 1.1, MIT Future License.
This repository is source-available today and converts to MIT two years after
each version is made available. See [`LICENSE`](./LICENSE).
