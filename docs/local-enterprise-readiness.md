# Local Enterprise Readiness

JR AutoRAG is positioned as a local enterprise RAG evaluation utility for IT, security, and application owners. It is intentionally local-first, with no required cloud dependency. Treat it like a client-owned assessment tool in the same operational category as local security testing utilities, not as a hosted multi-tenant SaaS platform.

Hosted enterprise SaaS deployment is out of scope for this repo. A hosted multi-tenant service would still need separate tenant isolation, hosted identity integration, durable audit storage, infrastructure hardening, and production incident operations.

## Supported Local Install Paths

| Path | Use when | Command |
| --- | --- | --- |
| Local source install | Developer or evaluator machine with Bun and uv | `bun install && bun run api:sync && bun run dev:all` |
| Auth-enabled interview demo | Live interview or client walkthrough | `bun run demo:interview` |
| Browser smoke | CI or repeatable interview-path regression | `bun run e2e:interview` |
| Standard container | Basic local container evaluation | `docker compose up --build` |
| Hardened local container | Client-owned workstation, jump host, or lab VLAN | `AUTORAG_API_KEYS=... docker compose -f docker-compose.enterprise.yml up --build` |
| Release artifact | Versioned source package with checksum | `bun run release:artifact` |

The API Python project is currently source-install oriented through `uv sync --project api`. Do not claim `pipx install` or `uv tool install` support until the API is intentionally converted into an installable package with console entrypoints.

## Provider Compatibility

JR AutoRAG can run without a cloud model provider for retrieval and grounded context summaries. Generated answers can use local or client-owned OpenAI-compatible endpoints:

- Ollama on localhost or a private client network.
- LM Studio on localhost or a private client network.
- vLLM or another private OpenAI-compatible endpoint.
- Public OpenAI-compatible providers only when the deployment profile and written engagement policy allow external model calls.

Use `deployment_profile=local_only` for offline workstation runs. Use `deployment_profile=client_safe` for client-owned network runs. In client-safe mode, provider URLs must be localhost, loopback, private-network IPs, or client-owned internal hostnames. Link-local provider targets are blocked by default and require `AUTORAG_ALLOW_LINK_LOCAL_PROVIDER=true`.

## Local Safety Rails

- `AUTORAG_EXPOSE=true` requires API-key authentication.
- Exposed mode disables `/docs`, `/redoc`, and `/openapi.json`.
- Destructive cache, raw trace bundle, config mutation, model download, and artifact build routes require admin scope.
- Non-admin trace lists are filtered to traces created by the same API key.
- Missing document ACLs fail closed when auth is enabled.
- Non-admin document lists return metadata and previews, not full document text.
- Client-safe policy rejects public providers, cloud backends, cloud OCR fallback, managed cloud hosting, and external model calls by default.
- RAGFuzz and dangerous poison-mode tests should stay disabled unless the engagement explicitly authorizes them.

## Evidence Handoff

Run the evidence bundle after the local install is configured and the client-readiness benchmark has a passing receipt:

```bash
bun run evidence:bundle
bun run handoff:gate -- evidence/install/<timestamp>-install-evidence
```

Archive the generated `SHA256SUMS`, `manifest.json`, `doctor.json`, `install-report.json`, `security-posture.json`, `config-policy.json`, supply-chain artifacts, secret-scan output, research architecture matrix, and client-readiness report with the engagement record.

## Operator Closeout

Before leaving a client environment:

- Export the redacted report pack and evidence bundle.
- Confirm whether the client wants the local `data/` volume handed off or deleted.
- Remove temporary uploads, poison-mode documents, trace bundles, and local caches that are not part of the handoff.
- Rotate or remove `AUTORAG_API_KEYS`, provider credentials, and `AUTORAG_VAULT_KEY`.
- Save cleanup proof with the engagement record when poison-mode or adversarial tests were run.

## Repeatable Gates

Run these gates before calling a local build enterprise-ready:

```bash
bun run api:lint
bun run api:test
bun run typecheck
bun test
bun run build
bun run e2e:interview
bash scripts/secret-scan.sh
bash scripts/research-architecture-check.sh
bash scripts/container-manifest-check.sh
bun run verify
bash scripts/release-gate.sh
```

Use `bash scripts/release-gate.sh --skip-container-smoke` only when Docker is unavailable, and record that skipped container smoke as a remaining release blocker until a Docker-enabled machine or CI runs it.
