# Baseline Evidence

Generated: 2026-02-12T14:29:39Z

## Context
```text
timestamp_utc=2026-02-12T14:28:52Z
git_commit=804100da38cb52481dad052e881da1e2b12148e6
git_branch=main
 M .github/workflows/ci.yml
 M Public/SECURITY.md
 M README.md
 M api/app/core/graph_rag.py
 M api/app/core/orchestrator.py
 M api/app/core/rate_limiter.py
 M api/app/core/security_middleware.py
 M api/app/routers/ragfuzz_audit.py
 M api/app/services.py
 M bun.lock
 M package-lock.json
 M package.json
 M src/App.tsx
 M src/components/features/AdvancedRAGSettings.tsx
 M src/components/features/ArtifactViewer.tsx
 M src/components/features/ChatInterface.tsx
 M src/components/features/EnterpriseStatusPanel.tsx
 M src/components/features/PipelineTimeline.tsx
 M src/components/features/ProviderCarousel.tsx
 M src/types.ts
?? SECURITY.md
?? api/app/core/milvus_store.py
?? audit/
```

## Runtime Versions
```text
bun 1.3.9
node v25.6.0
npm 11.8.0
python Python 3.14.3
pip pip 26.0 from /opt/homebrew/lib/python3.14/site-packages/pip (python 3.14)
```

## Output Hashes (SHA-256)

| File | SHA-256 |
|---|---|
| 00_context.txt | 9fb6fd4add6a0b9a4c585c49b8c997382d0fd6d0f7071d02c879fca8952c9a1a |
| 01_versions.txt | f4c8b376f4f6d0faaed990c1b07f77dab25eba27f056eed55da8b6681e1fe49d |
| 02_npm_tree.txt | d5b962f76c6cbbc8a88b2b194ec01f6b10fd6f8c805f2dc388eaf57e4302c7aa |
| 03_pip_freeze.txt | a48fbe3cdbd805602b886768b298a165ed7f80ed0f2b6642a0145d7dd86e652f |
| 10_typecheck.txt | 9f09acfe6102e6cc0175de6ffcaa3af20e61b5eabde18fd4cad61173bcecc90c |
| 11_bun_test.txt | 107fcbc6de156029cf6e08164d91aff289efad9d81165497976b508c455cf717 |
| 12_pytest.txt | f656bbe059ca586b10b00e31e54904adccc773308ac0e0bcbe30f352337fe64e |
| 13_ruff_statistics.txt | 729e3ff083231edcd3ac74e99e8379c509eec2f822dafcd94408ed3cea8d8f6a |
| 14_npm_audit.json | c65af33a47448cad75129390c693c4f94996749a0358e39d5ebe6d85433c428a |

## Command Outputs

### 10_typecheck.txt
```text
$ tsc --noEmit
exit_code=0
```

### 11_bun_test.txt
```text
bun test v1.3.9 (cf6cdbbb)

src/lib/utils.test.ts:
(pass) cn > merges only truthy class names [6.83ms]
(pass) cn > deduplicates classes with tailwind merge semantics [0.02ms]

 2 pass
 0 fail
 2 expect() calls
Ran 2 tests across 1 file. [84.00ms]
exit_code=0
```

### 12_pytest.txt
```text
============================= test session starts ==============================
platform darwin -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /Users/jonathanreed/Downloads/JR-AutoRAG/api
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.11.0, asyncio-1.3.0, langsmith-0.4.42, typeguard-4.4.4, hydra-core-1.3.2, cov-7.0.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 165 items

tests/api/test_endpoints.py ..                                           [  1%]
tests/test_advanced_integration.py ........................              [ 15%]
tests/test_binary_quantization.py ..........................             [ 31%]
tests/test_bq_retrieval.py .....................                         [ 44%]
tests/test_metrics_alignment.py ..                                       [ 45%]
tests/test_milvus_store.py .....................                         [ 58%]
tests/test_pipeline_stages.py .......................................... [ 83%]
                                                                         [ 83%]
tests/test_smoke.py .....                                                [ 86%]
tests/test_sota_final.py .....                                           [ 89%]
tests/test_vnext_expansion.py .................                          [100%]

============================= 165 passed in 19.80s =============================
exit_code=0
```

### 13_ruff_statistics.txt
```text
4270	W293  	[*] blank-line-with-whitespace
 183	UP037 	[*] quoted-annotation
 152	F401  	[ ] unused-import
 140	W291  	[*] trailing-whitespace
  77	I001  	[*] unsorted-imports
  55	UP045 	[*] non-pep604-annotation-optional
  25	UP017 	[*] datetime-timezone-utc
  22	UP035 	[*] deprecated-import
  16	SIM102	[ ] collapsible-if
  13	SIM108	[ ] if-else-block-instead-of-if-exp
  11	B905  	[ ] zip-without-explicit-strict
  11	SIM105	[ ] suppressible-exception
  10	C401  	[ ] unnecessary-generator-set
  10	F841  	[ ] unused-variable
   8	F541  	[*] f-string-missing-placeholders
   7	B007  	[ ] unused-loop-control-variable
   6	UP015 	[*] redundant-open-modes
   4	B023  	[ ] function-uses-loop-variable
   4	B904  	[ ] raise-without-from-inside-except
   4	E741  	[ ] ambiguous-variable-name
   3	E701  	[ ] multiple-statements-on-one-line-colon
   3	UP041 	[*] timeout-error-alias
   2	C416  	[ ] unnecessary-comprehension
   2	SIM103	[ ] needless-bool
   2	SIM110	[ ] reimplemented-builtin
   2	SIM114	[*] if-with-same-arms
   2	SIM117	[ ] multiple-with-statements
   1	B009  	[*] get-attr-with-constant
   1	B027  	[ ] empty-method-without-abstract-decorator
   1	C414  	[ ] unnecessary-double-cast-or-process
   1	E731  	[ ] lambda-assignment
   1	SIM113	[ ] enumerate-for-loop
   1	UP022 	[ ] replace-stdout-stderr
Found 5050 errors.
[*] 4342 fixable with the `--fix` option (623 hidden fixes can be enabled with the `--unsafe-fixes` option).
exit_code=1
```

### 14_npm_audit.json
```text
{
  "auditReportVersion": 2,
  "vulnerabilities": {},
  "metadata": {
    "vulnerabilities": {
      "info": 0,
      "low": 0,
      "moderate": 0,
      "high": 0,
      "critical": 0,
      "total": 0
    },
    "dependencies": {
      "prod": 172,
      "dev": 7,
      "optional": 11,
      "peer": 12,
      "peerOptional": 0,
      "total": 190
    }
  }
}
exit_code=0
```

## Dependency Snapshots

- NPM tree: `/Users/jonathanreed/Downloads/JR-AutoRAG/audit/evidence/raw/02_npm_tree.txt`
- Pip freeze: `/Users/jonathanreed/Downloads/JR-AutoRAG/audit/evidence/raw/03_pip_freeze.txt`
