# Security Guide for JR AutoRAG

This document covers security configuration, best practices, and deployment guidelines for running JR AutoRAG in production environments.

## Quick Start

By default, JR AutoRAG runs with **safe local defaults**:
- CORS allows only localhost origins
- Authentication is disabled (for local development)
- Rate limiting is disabled
- Server binds to localhost only

For production, enable security features via environment variables:

```bash
# Enable authentication
export AUTORAG_AUTH_ENABLED=true
export AUTORAG_API_KEYS="your-secret-key-1,your-secret-key-2"

# Enable rate limiting
export AUTORAG_RATE_LIMIT_ENABLED=true

# Configure CORS for your domain
export AUTORAG_ALLOWED_ORIGINS="https://app.yourdomain.com,https://admin.yourdomain.com"

# Start the API
uvicorn app.main:app --host 127.0.0.1 --port 8000
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
| `query` | Query endpoints (`/query/*`) |
| `ingest` | Document endpoints (`/documents/*`) |
| `admin` | Configuration and admin endpoints |
| `eval` | Evaluation endpoints (`/evaluation/*`) |

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
   pip install keyring
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

### Exposing to Network

To expose the API beyond localhost:

```bash
# Enable exposed mode
export AUTORAG_EXPOSE=true

# Configure allowed origins (REQUIRED when exposed)
export AUTORAG_ALLOWED_ORIGINS="https://yourdomain.com"

# Bind to all interfaces
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> ⚠️ **Warning**: Never expose the API without authentication enabled and proper CORS configuration.

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
- `AUTORAG_RATE_LIMIT_ENABLED`: Set to `true` to enable
- `AUTORAG_RATE_LIMIT_RPM`: Requests per minute (default: 60)
- `AUTORAG_RATE_LIMIT_BURST`: Burst capacity (default: 10)

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
- [ ] Enable rate limiting
- [ ] Configure TLS via reverse proxy
- [ ] Review and rotate API keys regularly
- [ ] Set up audit log monitoring
- [ ] Configure request size limits appropriately
- [ ] Enable PII detection if handling sensitive data
- [ ] Test with security scanning tools

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTORAG_AUTH_ENABLED` | `false` | Enable API key authentication |
| `AUTORAG_API_KEYS` | (none) | Comma-separated API keys |
| `AUTORAG_ALLOWED_ORIGINS` | localhost only | CORS allowed origins |
| `AUTORAG_EXPOSE` | `false` | Allow non-localhost binding |
| `AUTORAG_RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `AUTORAG_RATE_LIMIT_RPM` | `60` | Requests per minute |
| `AUTORAG_MAX_REQUEST_SIZE` | `52428800` | Max request body (50MB) |
| `AUTORAG_VAULT_KEY` | (auto) | Encryption key for secrets vault |

## Reporting Security Issues

If you discover a security vulnerability, please report it privately before public disclosure.

Contact: [Add your security contact here]
