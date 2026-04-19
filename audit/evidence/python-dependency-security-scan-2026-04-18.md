# Python Dependency and Security Scan

Generated: 2026-04-18T22:25:00Z

## Scope

- Repository: `JR-AutoRAG`
- Backend manifest: `api/requirements.txt`
- Runtime target: Python 3.11
- Dependency scan: `uv pip compile` plus `pip-audit`
- Source security scan: `bandit -r api/app`
- Regression tests: FastAPI/backend pytest suite in a fresh temporary Python 3.11 environment

## Result

- Python dependency audit: clean after dependency patches.
- High-severity Bandit findings: clean after replacing non-cryptographic MD5 fingerprints with SHA-256.
- Full backend tests: passing.

## Dependency Changes

The original requirements graph was not resolvable because `langextract[openai]>=1.1.1` depends on `google-genai>=1.39.0`, which requires `httpx>=0.28.1`, while JR-AutoRAG pinned `httpx==0.27.2`.

Patched packages:

| Package | Previous | Updated | Reason |
|---|---:|---:|---|
| `fastapi` | `0.115.0` | `0.136.0` | Allows fixed `starlette` transitive version. |
| `httpx` | `0.27.2` | `0.28.1` | Required by `google-genai` via `langextract[openai]`. |
| `python-multipart` | `0.0.9` | `0.0.26` | Fixes multipart parser advisories. |
| `pypdf` | `4.3.1` | `6.10.2` | Fixes crafted-PDF denial-of-service advisories. |
| `pytest` | `8.3.3` | `9.0.3` | Fixes local `/tmp/pytest-of-{user}` advisory. |
| `pytest-asyncio` | `0.24.0` | `1.3.0` | Restores compatibility with pytest 9. |

## Evidence

| Evidence | Result |
|---|---|
| `raw/15_uv_pip_compile.txt` | Resolved 140 packages, exit 0. |
| `raw/16_requirements_lock_snapshot.txt` | Resolved Python 3.11 dependency snapshot used for audit. |
| `raw/17_pip_audit_resolved.json` | 140 dependencies scanned, 0 vulnerable packages. |
| `raw/17_pip_audit_resolved.exit` | `exit_code=0`. |
| `raw/18_bandit_api_app.json` | 43 residual findings: 15 medium, 28 low, 0 high. |
| `raw/18_bandit_api_app.exit` | `exit_code=1`; Bandit exits non-zero for remaining medium/low findings. |
| `raw/19_pytest_dependency_patch.txt` | Targeted dependency regression suite: 35 passed. |
| `raw/20_pytest_full_dependency_patch.txt` | Full backend suite: 181 passed. |
| `raw/21_pytest_hash_security_patch.txt` | Hash/security regression suite: 72 passed. |

## Residual Security Items

Bandit still reports medium/low findings that should be handled in a follow-up hardening ticket:

- `B301`/`B403`: pickle-backed binary vector store persistence should be restricted to trusted local artifacts or migrated to a safer serialization format.
- `B314`/`B405`: XML parsing in ingestion should move to `defusedxml` before accepting untrusted XML-heavy document formats.
- `B615`: Hugging Face `from_pretrained()` calls should pin model revisions for repeatable and safer model downloads.
- `B603`: subprocess OCR/doc conversion path should keep strict argument construction and add explicit allowlist documentation.
- `B110`/`B112`/`B311`: low-severity exception handling and non-security randomness findings remain.

