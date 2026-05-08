# JR AutoRAG

JR AutoRAG is a local-first Retrieval-Augmented Generation workbench for document-grounded AI systems. It pairs a Bun and React admin console with a FastAPI backend so you can ingest documents, configure local LLM providers, ask grounded questions, inspect citations, and review pipeline traces.

The product is built for two use cases:

- Normal product use: persistent local documents, provider configuration, retrieval, citations, and traces.
- Evaluator demo use: a disposable local corpus that gets an evaluator to a credible first answer quickly.

## What It Shows

- Local-first document ingestion for PDF, DOCX, Markdown, TXT, and pasted text.
- Provider discovery for Ollama and LM Studio, with manual OpenAI-compatible profiles.
- Hybrid retrieval with dense vectors, sparse matching, reranking, evidence contracts, and answerability calibration.
- Visible RAG controls for grounded vs open querying, presets, query scope, and trace inspection.
- Quality cockpit surfaces for parser preview, recommendations, experiment runs, and evaluation signals.
- Security defaults for localhost, optional API-key auth, request limits, safe CORS, and local-only deployment policy.

## Architecture

```text
+----------------+      +------------------+      +-----------------------------+
| Bun + React UI | ---> | FastAPI Backend  | ---> | Ollama / LM Studio / APIs   |
+----------------+      +------------------+      +-----------------------------+
         |                         |
         |                         +--> Documents, indexes, traces, metrics
         +--> shadcn-based admin console
```

## Requirements

- Bun 1.3 or newer.
- Python 3.11 or newer.
- Optional local LLM runtime: Ollama on `http://localhost:11434` or LM Studio on `http://localhost:1234`.
- Optional OCR tooling for scanned PDFs:
  - macOS: `brew install tesseract poppler`
  - Ubuntu or Debian: `sudo apt-get install tesseract-ocr poppler-utils`
  - Fedora: `sudo dnf install tesseract poppler-utils`

## Quickstart

```bash
bun install
cd api && python3 -m pip install -r requirements.txt
cd ..
bun run doctor
bun run dev:all
```

Default URLs:

- Admin console: <http://localhost:3000>
- API: <http://localhost:8000>

If port 3000 is busy, run the frontend manually on another port:

```bash
PORT=3001 BUN_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 bun --hot src/index.ts
```

Run the local install doctor any time an operator changes machines, ports, OCR tooling, auth settings, or model runtimes:

```bash
bun run doctor
bun run doctor -- --json
```

The doctor checks Bun, Python, core API imports, OCR tools, API and web ports, data directory write access, auth environment consistency, security posture, and local Ollama or LM Studio availability. Warnings do not block local document search, but failed checks should be fixed before a client install.

After the API is running, collect the full install evidence bundle:

```bash
bun run evidence:bundle
```

The bundle is written under `evidence/install/` by default and includes `doctor.json`, `readyz.json`, `config-policy.json`, `security-posture.json`, `install-report.json`, `research-architecture.md`, `manifest.json`, and `SHA256SUMS`. Set `JR_EVIDENCE_API_BASE_URL` or pass `--api-base-url` when collecting evidence from a non-default API origin.

## Normal Product Workflow

Use this path to verify the real product, not only the demo seed:

1. Start the API and UI with `bun run dev:all`.
2. Open Configuration.
3. Apply a local provider such as Ollama. The UI should show available models and save planner, gatherer, and generator selections.
4. Open Documents.
5. Upload a Markdown, TXT, PDF, or DOCX file.
6. Confirm the document status becomes ready.
7. Open Query.
8. Ask a grounded question about the uploaded document.
9. Inspect the answer citations, source list, trace, and quality signals.

The product can answer without a configured provider by returning a grounded context summary, but provider setup is required for normal generated answers.

## Evaluator Demo Mode

Demo mode uses a temporary local data directory when no explicit data directory is set. It is meant for interviews, walkthroughs, and fast evaluation. Data is disposable after the app exits.

```bash
JR_DEMO_MODE=1 bun run dev:all
```

Then use the Configuration onboarding panel:

1. Click Load Demo Corpus.
2. Ask one of the example questions.
3. Inspect citations and the Why this answer panel.
4. Open Quality to review recommendations and experiment surfaces.

Demo mode also relaxes local rate limiting unless `AUTORAG_RATE_LIMIT_ENABLED` is explicitly set, so browser reloads and walkthrough polling do not interrupt the evaluator path.

## Data Storage

Normal runs store local state under ignored data directories:

- `data/`
- `api/data/`

Set `JR_DATA_DIR` when you want an explicit state location:

```bash
JR_DATA_DIR=/path/to/local-state bun run dev:all
```

Do not commit data directories, `.env` files, API keys, provider secrets, traces containing sensitive content, or client documents.

## Configuration

Common environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BUN_PUBLIC_API_BASE_URL` | `http://127.0.0.1:8000` on local UI hosts | Frontend API target |
| `VITE_API_BASE_URL` | Same as above | Vite-compatible API target |
| `JR_DEMO_MODE` | unset | Use disposable demo storage when set to `1`, `true`, or `yes` |
| `JR_DATA_DIR` | unset | Explicit local data directory |
| `AUTORAG_AUTH_ENABLED` | `false` | Require API keys |
| `AUTORAG_API_KEYS` | unset | Comma-separated API keys |
| `AUTORAG_ALLOWED_ORIGINS` | localhost defaults | CORS allow list |
| `AUTORAG_EXPOSE` | `false` | Allow non-localhost binding |
| `AUTORAG_RATE_LIMIT_ENABLED` | `true`, `false` in demo mode | Enable request rate limiting |
| `AUTORAG_RATE_LIMIT_RPM` | `100` | Requests per minute |
| `AUTORAG_RATE_LIMIT_BURST` | `20` | Burst capacity |

## Security Defaults

- Binds to localhost by default.
- Allows only local UI origins unless configured.
- Supports optional API-key authentication.
- Reports redacted install posture at `GET /security/posture`.
- Exports a redacted client handoff report at `GET /install/report`.
- Blocks `/docs`, `/redoc`, and `/openapi.json` when `AUTORAG_EXPOSE=true`.
- Applies request size limits, route timeouts, and security headers.
- Uses local-only deployment policy by default, rejecting public cloud providers unless the deployment profile allows them.
- Keeps client-adjacent policy fields explicit, including retention, external model calls, PII redaction, and operator review.

See [Public/SECURITY.md](./Public/SECURITY.md) for production deployment guidance.

## API Surfaces

Useful endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | API health |
| `GET /readyz` | Runtime readiness with degraded states |
| `GET /security/posture` | Redacted install security posture |
| `GET /install/report` | Redacted client handoff report with readiness, posture, corpus, evaluation, and artifact evidence |
| `GET /config` | Current product configuration |
| `PUT /config` | Save provider, retrieval, and deployment settings |
| `GET /providers/local` | Discover Ollama and LM Studio |
| `POST /documents/upload` | Upload PDF, DOCX, Markdown, or TXT |
| `POST /documents/text` | Ingest pasted text |
| `POST /query` | Ask a normal non-streaming query |
| `POST /query/stream` | Ask a streaming query |
| `POST /query/cancel` | Cancel an in-flight stream |
| `GET /onboarding` | Read onboarding state and example prompts |
| `POST /onboarding/demo/seed` | Seed the disposable demo corpus |
| `DELETE /onboarding/demo` | Remove demo documents |
| `GET /config/recommendations` | Quality cockpit recommendations |
| `GET /experiments` | Preset comparison and experiment runs |
| `GET /evaluation/runs` | Evaluation run history |

## Document Ingestion

| Input type | Handling |
| --- | --- |
| PDF | Text extraction with OCR fallback for scans |
| DOC/DOCX | Native text extraction through a temporary parse path |
| Markdown | Markdown-aware text extraction |
| TXT | UTF-8 text extraction with tolerant decoding |
| Pasted text | Direct ingestion from the Documents panel |

Uploads trigger indexing and retrieval rebuilds. Deleting a document also rebuilds the index.

## Retrieval Presets

| Preset | Best for |
| --- | --- |
| Turbo | Lowest-latency checks |
| Fast | Quick answers with modest context |
| Balanced | Default workbench setting |
| Thorough | Deep research and broader recall |
| Ultra Accurate | Maximum context and stricter quality posture |

The UI surfaces related controls for hybrid search, reranking, RAPTOR, graph retrieval, evidence contracts, HyDE, and Self-RAG critique where supported by the current backend settings.

See [docs/architecture/research-backed-rag-architecture.md](./docs/architecture/research-backed-rag-architecture.md) for the current research-to-implementation matrix used in client evidence bundles.

## Verification

Run these before sharing a branch:

```bash
bun run doctor:test
bun run typecheck
bun test
bun run build
.venv/bin/python -m ruff check api/app api/tests --statistics
cd api && PYTHONPATH=. ../.venv/bin/pytest -q
git diff --check
```

Manual smoke test:

```bash
# Start the app first.
bun run dev:all

# In the UI, apply Ollama or another provider.
# Upload a real document from the Documents tab.
# Ask a grounded question from the Query tab.
# Confirm citations, sources, and traces render.
```

API smoke test:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/providers/local
curl http://127.0.0.1:8000/documents
```

## Troubleshooting

| Issue | Fix |
| --- | --- |
| UI cannot reach API | Confirm FastAPI is on port 8000 and set `BUN_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`. |
| Provider list is empty | Start Ollama or LM Studio, then rescan providers. |
| Query returns context summary only | Apply a provider in Configuration. |
| PDF upload has no text | Install Poppler and Tesseract, then restart the API. |
| CORS error on a different local port | Add the origin with `AUTORAG_ALLOWED_ORIGINS`. |
| 429 responses during local walkthrough | Use `JR_DEMO_MODE=1` or raise `AUTORAG_RATE_LIMIT_RPM` for local testing. |

## Deployment

1. Build the UI with `bun run build`.
2. Serve `dist/` behind a static server.
3. Deploy the FastAPI app behind HTTPS.
4. Set `BUN_PUBLIC_API_BASE_URL` to the API origin at build/runtime.
5. Enable `AUTORAG_AUTH_ENABLED=true`.
6. Set `AUTORAG_API_KEYS`.
7. Set exact `AUTORAG_ALLOWED_ORIGINS`.
8. Run `bun run doctor -- --json` and fix any failed `security_posture` check.
9. Verify `GET /security/posture` returns no failed checks.
10. Run `bun run evidence:bundle` and keep the generated directory with the client handoff evidence.
11. Persist `data/` or set `JR_DATA_DIR` to durable storage.
12. Review [Public/SECURITY.md](./Public/SECURITY.md) before exposing the API.

## License

Licensed under the Functional Source License, Version 1.1, MIT Future License.
This repository is source-available today and converts to MIT two years after each version is made available. See [LICENSE](./LICENSE).
