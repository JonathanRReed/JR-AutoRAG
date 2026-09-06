# JR AutoRAG

Import documents, ask questions about them, and inspect the citations and retrieval traces. JR AutoRAG combines a React console with a FastAPI backend and supports local Ollama, LM Studio, and configured OpenAI-compatible providers.

Normal use keeps documents and indexes in local storage. Demo mode uses a disposable corpus. Neither a successful demo nor a running model establishes client readiness.

## Start locally

Requires Bun 1.4+, Python 3.11+, and uv 0.12+.

```bash
bun install
bun run api:sync
bun run doctor
bun run dev:all
```

Open the console at `http://localhost:3000`; the API uses `http://localhost:8000`. Ollama normally uses port `11434`, and LM Studio uses `1234`.

`api:sync` includes ML extras for Docling and local embeddings. Scanned PDFs also need Tesseract and Poppler:

```bash
# macOS
brew install tesseract poppler
# Ubuntu or Debian
sudo apt-get install tesseract-ocr poppler-utils
# Fedora
sudo dnf install tesseract poppler-utils
```

For another frontend port:

```bash
PORT=3001 BUN_PUBLIC_BROWSER_API_BASE_URL=http://127.0.0.1:8000 bun --hot src/index.ts
```

## Ask a document question

In Configuration, apply a provider and select planner, gatherer, and generator models. In Documents, upload a PDF, DOCX, Markdown, or text file, or paste text. Wait for ready status, then ask a grounded question in Query. Inspect its citations, source list, trace, and quality checks.

Without a configured provider, the app returns a retrieved-context summary rather than a generated answer. Uploading or deleting documents rebuilds retrieval indexes. PDF parsing falls back to OCR when needed; Word files use a temporary parsing path.

Retrieval combines dense vectors, sparse matching, reranking, and evidence checks. Presets range from Turbo for low latency to Ultra Accurate for more context and stricter checks, with Fast, Balanced, and Thorough between them. Backend-supported controls include query scope, grounded or open querying, RAPTOR, graph retrieval, HyDE, and Self-RAG critique. See [UPGRADE-2026.md](UPGRADE-2026.md).

## Disposable demo

```bash
JR_DEMO_MODE=1 bun run dev:all
```

In Configuration, select Load Demo Corpus, ask an example question, and inspect citations and Why this answer. Quality contains recommendations and experiments.

Without an explicit data directory, demo storage is temporary and disposable after exit. Demo mode also relaxes local rate limits unless `AUTORAG_RATE_LIMIT_ENABLED` is set explicitly. Do not use it as a production configuration.

## Storage and settings

Normal runs use ignored `data/` and `api/data/` directories. Set a durable location with:

```bash
JR_DATA_DIR=/path/to/local-state bun run dev:all
```

Do not commit documents, data directories, `.env` files, credentials, or traces containing sensitive text.

| Variable | Default and purpose |
| --- | --- |
| `BUN_PUBLIC_BROWSER_API_BASE_URL` | Browser API target; `http://127.0.0.1:8000` on local UI hosts |
| `VITE_BROWSER_API_BASE_URL` | Vite-compatible browser target |
| `BUN_PUBLIC_API_BASE_URL` | Server target for the limited legacy `/api/*` proxy; same local default |
| `VITE_API_BASE_URL` | Vite-compatible proxy target |
| `JR_DEMO_MODE` | Unset; `1`, `true`, or `yes` enables disposable demo mode |
| `JR_DATA_DIR` | Unset; explicit local storage path |
| `AUTORAG_AUTH_ENABLED` | `false`; requires API keys when enabled |
| `AUTORAG_API_KEYS` | Unset; comma-separated keys |
| `AUTORAG_ALLOWED_ORIGINS` | Localhost CORS defaults |
| `AUTORAG_EXPOSE` | `false`; permits non-localhost binding when enabled |
| `AUTORAG_RATE_LIMIT_ENABLED` | `true`, or `false` in demo mode |
| `AUTORAG_RATE_LIMIT_RPM` | `100` requests per minute |
| `AUTORAG_RATE_LIMIT_BURST` | `20` |

The API binds locally by default, limits request sizes and duration, and applies security headers. Local-only deployment policy rejects public cloud providers unless the selected profile permits them. Retention, external model calls, redaction, and operator review remain explicit settings.

`AUTORAG_EXPOSE=true` blocks `/docs`, `/redoc`, and `/openapi.json`; it does not replace authentication. Review [SECURITY.md](SECURITY.md) before exposing the API.

## API reference

| Endpoint | Use |
| --- | --- |
| `GET /healthz`, `GET /readyz` | Health and readiness, including degraded states |
| `GET /security/posture`, `GET /install/report` | Redacted security and handoff reports |
| `GET /config`, `PUT /config` | Read or save configuration |
| `GET /providers/local` | Discover Ollama and LM Studio |
| `POST /documents/upload`, `POST /documents/text` | Import files or text |
| `POST /query`, `POST /query/stream` | Ask a question |
| `POST /query/cancel` | Cancel a stream |
| `GET /onboarding` | Onboarding and example prompts |
| `POST /onboarding/demo/seed`, `DELETE /onboarding/demo` | Add or remove demo documents |
| `GET /config/recommendations`, `GET /experiments` | Quality recommendations and experiments |
| `GET /evaluation/runs` | Evaluation history |

## Verify

```bash
bun run verify
bash scripts/release-gate.sh
```

`verify` runs API lint and tests, evidence and handoff tests, architecture checks, TypeScript, frontend tests, and the web build. The release script adds the install doctor, smoke tests, container manifest and build checks, secret scanning, supply-chain evidence, and whitespace checks.

Run the release script through Bash because its install smoke test starts a nested Bun server. `--skip-container-smoke` permits a local run without Docker, but the skipped check remains a release blocker until it runs elsewhere.

Focused checks:

```bash
bun run doctor:test
bash scripts/install-smoke.test.sh
bash scripts/container-manifest-check.sh
bun run research:check
bun run container:smoke
bash scripts/secret-scan.sh
bun run supply-chain
git diff --check
```

The container smoke builds both images, imports the API, checks `/healthz` and `/readyz`, and verifies the web shell and assets. The default API image uses core dependencies and `UV_TORCH_BACKEND=cpu`; choose a GPU-specific image only for a deployment that needs it.

## Client handoff

Run `bun run doctor` after changing machines, ports, OCR tools, authentication, or model runtimes. It checks dependencies, imports, ports, writable storage, authentication settings, and local providers. Fix failed checks before installation.

For deployment, build and serve `dist/`, put FastAPI behind HTTPS, set the browser API origin, enable `AUTORAG_AUTH_ENABLED`, supply `AUTORAG_API_KEYS`, and allow only exact frontend origins. Persist the data directory.

Run the Client Readiness benchmark in Quality, or use `POST /evaluation/golden-sets/builtins` followed by `POST /evaluation/batch/client_readiness`. Then collect and validate the evidence:

```bash
bun run doctor -- --json
JR_EVIDENCE_API_KEY="${AUTORAG_API_KEYS%%,*}" bun run evidence:bundle
bun run handoff:gate -- evidence/install/<timestamp>-install-evidence
```

Use `JR_EVIDENCE_API_BASE_URL` or `--api-base-url` for a non-default API. The bundle goes to `evidence/install/` and includes installation, security, supply-chain, dependency, evaluation, and readiness reports plus a manifest and hashes. Authenticated collection uses `X-API-Key`.

The handoff gate requires valid hashes, all required artifacts, ready installation status, `client_ready` security, and a passing `client_readiness` receipt. That receipt must cover mixed formats, prompt injection, poisoned documents, extraction refusal, abstention, and binary, agentic, and graph retrieval. Demo or incomplete bundles should fail. Keep the passing bundle with the handoff records.

## Troubleshooting

| Problem | Check |
| --- | --- |
| UI cannot reach API | Port 8000 and `BUN_PUBLIC_BROWSER_API_BASE_URL` |
| No providers | Start Ollama or LM Studio, then rescan |
| Context summary instead of an answer | Apply a provider in Configuration |
| Scanned PDF has no text | Install Poppler and Tesseract, then restart the API |
| CORS error on another local port | Add the exact origin to `AUTORAG_ALLOWED_ORIGINS` |
| Rate limits during a demo | Use demo mode or adjust the local test limit |

## License

Functional Source License 1.1, MIT Future License. Each version converts to MIT two years after it is made available. See [LICENSE](LICENSE).
