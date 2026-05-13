# Security Guide for JR AutoRAG

This document covers security configuration, best practices, and deployment guidelines for running JR AutoRAG in production environments.

## Quick Start

By default, JR AutoRAG runs with **safe local defaults**:
- CORS allows only localhost origins
- Authentication is disabled (for local development)
- Rate limiting is enabled (`100` req/min, burst `20`)
- Server binds to localhost only

For production, enable security features via environment variables:

```bash
# Enable authentication
export AUTORAG_AUTH_ENABLED=true
export AUTORAG_API_KEYS="your-secret-key-1,your-secret-key-2"

# Configure CORS for your domain
export AUTORAG_ALLOWED_ORIGINS="https://app.yourdomain.com,https://admin.yourdomain.com"

# Start the API
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Authentication

### API Key Authentication

JR AutoRAG uses API key authentication with SHA-256 hashing. Keys are never stored in plaintext.

**Environment Variables:**
- `AUTORAG_AUTH_ENABLED`: Set to `true` to enable authentication
- `AUTORAG_API_KEYS`: Comma-separated list of valid API keys

**Using API Keys:**

```bash
# Include in request header
curl -H "X-API-Key: your-api-key" http://localhost:8000/query/stream
```

**Generating API Keys:**

```python
import secrets
# Generate a secure random key
api_key = secrets.token_urlsafe(32)
print(api_key)  # e.g., "Ak7_xP9m2Lq..."
```

### Role-Based Access Control (RBAC)

API keys have associated scopes that control access:

| Scope | Access |
|-------|--------|
| `read` | Query/monitoring/read-only endpoints |
| `write` | Document ingest/update/delete and write actions |
| `admin` | Config/admin/ragfuzz/cache/artifact build controls |

By default, keys from `AUTORAG_API_KEYS` have all scopes.

## Secrets Management

Never store API keys (OpenAI, Anthropic, etc.) in plain configuration files.

### Options (in order of preference):

1. **Environment Variables** (Recommended for containers)
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. **OS Keychain** (Recommended for local development)
   ```bash
   cd api && uv pip install keyring
   ```
   Keys are automatically stored in:
   - macOS: Keychain Access
   - Windows: Windows Credential Store
   - Linux: Secret Service (GNOME Keyring, KWallet)

3. **Encrypted Vault** (Fallback)
   
   If keychain isn't available, secrets are stored in an encrypted file at `data/secrets.vault`.
   
   Set a custom encryption key:
   ```bash
   export AUTORAG_VAULT_KEY="your-encryption-passphrase"
   ```

## Network Security

### Safe Defaults

By default, JR AutoRAG:
- Binds to `127.0.0.1` (localhost only)
- Allows CORS only from `http://localhost:3000` and `http://localhost:5173`

## Client-Safe Deployment Profile

Use `deployment_profile=client_safe` for consulting engagements, regulated document reviews, or any client-adjacent assessment where JR AutoRAG must run in a local or client-owned boundary.

Client-safe mode is intentionally narrower than general production hosting:

- Runtime providers must use localhost, loopback, private-network IPs, or client-owned internal hostnames such as `.local`, `.lan`, or `.internal`.
- Public cloud model endpoints are rejected by config validation.
- Cloud backends are rejected.
- Managed cloud hosting is not allowed by the default data policy.
- External model calls are not allowed by the default data policy.
- Cloud OCR fallback is rejected.
- Reports should be exported redacted by default.

Recommended environment and config defaults:

```bash
export AUTORAG_AUTH_ENABLED=true
export AUTORAG_API_KEYS="generated-client-engagement-key"
export AUTORAG_ALLOWED_ORIGINS="https://autorag.client.internal"
export AUTORAG_EXPOSE=true
export AUTORAG_RATE_LIMIT_ENABLED=true
export AUTORAG_RAGFUZZ_ENABLED=false
export AUTORAG_PII_REDACT=true
```

Set the application config:

```json
{
  "deployment_profile": "client_safe",
  "provider": {
    "name": "Client Ollama",
    "base_url": "http://10.0.0.5:11434",
    "generator_model": "llama3.1"
  },
  "data_policy": {
    "classification": "client_confidential",
    "storage_boundary": "client_owned",
    "managed_cloud_hosting_allowed": false,
    "external_model_calls_allowed": false,
    "pii_redaction_required": true,
    "document_retention_days": 30,
    "trace_retention_days": 14,
    "report_export_mode": "redacted_by_default",
    "client_handoff_required": true,
    "operator_review_required": true
  }
}
```

Before any client run, verify the live policy:

```bash
curl -H "X-API-Key: ${AUTORAG_API_KEYS%%,*}" http://localhost:8000/config/policy
curl -H "X-API-Key: ${AUTORAG_API_KEYS%%,*}" http://localhost:8000/security/posture
curl -H "X-API-Key: ${AUTORAG_API_KEYS%%,*}" http://localhost:8000/install/report
bun run doctor -- --json
bun run evidence:bundle
```

The policy response includes `deployment_profile`, `data_policy`, and `guardrails`; the security posture response includes auth, CORS, exposure, docs, headers, and rate-limit checks. The install report combines readiness, posture, corpus state, evaluation receipts, retrieval artifacts, and redaction metadata. The evidence bundle saves these live responses, install smoke output, container manifest output, secret-scan output, supply-chain SBOM and audit output, the research-backed architecture matrix, hashes, and a manifest. Failed `security_posture` doctor checks must be fixed before a client-network install.

## Client Data Policy

For client-safe use, treat all ingested documents, prompts, retrieved chunks, traces, exports, and generated reports as `client_confidential` unless the client classifies them lower in writing.

Default handling rules:

- Store documents, indexes, traces, config, and audit logs only under the local or client-owned `data/` volume.
- Do not copy `data/`, traces, vector indexes, prompt logs, or exports into Hello.World managed cloud storage without written approval for the engagement.
- Keep document artifacts for no more than 30 days by default.
- Keep query traces for no more than 14 days by default.
- Export reports with redaction enabled by default; full exports require client approval and operator review.
- Redact PII before durable report handoff whenever the report does not require exact sensitive values.
- Rotate API keys at engagement close and remove any provider credentials from the secrets vault when handoff is complete.
- Delete or hand off `data/` at closeout according to the statement of work.

Minimum closeout checklist:

- [ ] Export final redacted report and remediation backlog.
- [ ] Confirm whether the client wants the local `data/` volume handed off or deleted.
- [ ] Delete local temporary upload files and unneeded trace bundles.
- [ ] Rotate or remove `AUTORAG_API_KEYS`, provider keys, and `AUTORAG_VAULT_KEY`.
- [ ] Save `/config/policy` output, the evidence bundle, SBOM, dependency audit, and test evidence with the engagement record.

### Exposing to Network

To expose the API beyond localhost:

```bash
# Enable exposed mode
export AUTORAG_EXPOSE=true

# REQUIRED in exposed mode: enable authentication first
export AUTORAG_AUTH_ENABLED=true
export AUTORAG_API_KEYS="your-secret-key"

# Configure allowed origins (REQUIRED when exposed)
export AUTORAG_ALLOWED_ORIGINS="https://yourdomain.com"

# Bind to all interfaces
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Warning: Never expose the API without authentication enabled and proper CORS configuration.

When `AUTORAG_EXPOSE=true`, JR AutoRAG will refuse non-public requests unless `AUTORAG_AUTH_ENABLED=true`.
Interactive API docs at `/docs`, `/redoc`, and `/openapi.json` are disabled in exposed mode.

### TLS/HTTPS

JR AutoRAG does not handle TLS directly. Use a reverse proxy.

#### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    
    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for SSE
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts for long-running queries
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

#### Caddy Configuration (Simpler)

```caddyfile
api.yourdomain.com {
    reverse_proxy localhost:8000
}
```

Caddy automatically handles TLS certificate provisioning via Let's Encrypt.

## Rate Limiting

Rate limiting prevents abuse and ensures fair usage.

**Environment Variables:**
- `AUTORAG_RATE_LIMIT_ENABLED`: Set to `true` to enable (default: `true`)
- `AUTORAG_RATE_LIMIT_RPM`: Requests per minute (default: 100)
- `AUTORAG_RATE_LIMIT_BURST`: Burst capacity (default: 20)

Rate limits are applied per API key (if authenticated) or per IP address.

**Response Headers:**
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Limit`: Total requests allowed per minute
- `Retry-After`: Seconds until next request allowed (when limited)

## Request Limits

**Environment Variables:**
- `AUTORAG_MAX_REQUEST_SIZE`: Maximum request body size in bytes (default: 50MB)

**Per-Route Timeouts:**

| Endpoint | Timeout |
|----------|---------|
| `/query/stream` | 300s |
| `/documents/upload` | 600s |
| `/evaluation/run` | 900s |
| Default | 60s |

## Audit Logging

All operations are logged to `data/audit/` in JSONL format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "query",
  "user_id": "Ak7_xP9m...",
  "ip_address": "192.168.1.100",
  "details": {
    "query": "What is...",
    "documents_accessed": ["doc1", "doc2"]
  },
  "success": true,
  "duration_ms": 1234
}
```

**Logged Actions:**
- `query`: Query operations
- `ingest`: Document ingestion
- `delete`: Document deletion
- `config_change`: Configuration changes
- `auth_success` / `auth_failure`: Authentication attempts
- `system`: Startup/shutdown events

## Security Headers

All responses include security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'none'` (for API responses)

## Prompt Injection Defense

JR AutoRAG includes defenses against prompt injection attacks:

1. **Pattern Detection**: Known injection patterns are detected and logged
2. **Content Sanitization**: Optional sanitization at ingest time
3. **Untrusted Data Boundaries**: Retrieved content is wrapped with clear boundaries

Enable strict sanitization:
```python
from app.core.prompt_guard import sanitize_at_ingest

sanitized_content, attempts = sanitize_at_ingest(
    content=raw_document,
    source="uploaded_document",
    wrap_delimiters=True
)
```

## Production Checklist

Before deploying to production:

- [ ] Enable authentication (`AUTORAG_AUTH_ENABLED=true`)
- [ ] Configure strong API keys
- [ ] Set appropriate CORS origins
- [ ] Configure TLS via reverse proxy
- [ ] Review and rotate API keys regularly
- [ ] Set up audit log monitoring
- [ ] Configure request size limits appropriately
- [ ] Enable PII detection if handling sensitive data
- [ ] Test with security scanning tools
- [ ] For client work, use `deployment_profile=client_safe` and save a `bun run evidence:bundle` output directory

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTORAG_AUTH_ENABLED` | `false` | Enable API key authentication |
| `AUTORAG_API_KEYS` | (none) | Comma-separated API keys |
| `AUTORAG_ALLOWED_ORIGINS` | localhost only | CORS allowed origins |
| `AUTORAG_EXPOSE` | `false` | Allow non-localhost binding |
| `AUTORAG_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `AUTORAG_RATE_LIMIT_RPM` | `100` | Requests per minute |
| `AUTORAG_RATE_LIMIT_BURST` | `20` | Burst capacity |
| `AUTORAG_MAX_REQUEST_SIZE` | `52428800` | Max request body (50MB) |
| `AUTORAG_VAULT_KEY` | (auto) | Encryption key for secrets vault |
| `AUTORAG_RAGFUZZ_ENABLED` | `true` in dev, `false` in prod | Enable RAGFuzz audit endpoints |
| `AUTORAG_RAGFUZZ_SECRET` | (none) | Shared secret for `/rag/audit/*` endpoints (required in prod) |

## Reporting Security Issues

If you discover a security vulnerability, please report it privately before public disclosure.

Contact: [Add your security contact here]
