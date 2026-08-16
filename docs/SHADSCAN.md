# Shadscan Review Workflow

JR AutoRAG is a direct Shadscan target. The frontend is React, uses a shadcn `components.json` registry, Radix primitives, Tailwind, and a task-heavy admin interface with forms, drawers, configuration, document state, query results, and evaluation surfaces.

The repository keeps Shadscan read-only and uninstalled. The package scripts pin version `0.17.0` so the report does not change merely because a new scanner release appeared.

## Source audit

```bash
bun run audit:shadscan
```

The command emits the versioned JSON report. It does not build the app, execute repository code through Shadscan, or change files.

Generate a neutral coding-agent handoff from the same rules:

```bash
bun run audit:shadscan:prompt
```

Review every finding before editing. A Shadscan score is not a product requirement by itself. Keep decisions that fit JR AutoRAG's actual information model and waive rules that assume a different product.

## Rendered overflow check

Start the real frontend first:

```bash
bun run dev:all
```

Then run the separate browser check:

```bash
bunx @shadscan/cli@0.17.0 \
  --check-ui http://127.0.0.1:3000 \
  --no-interactive \
  --no-roast
```

This rendered mode checks document-level horizontal overflow at Shadscan's fixed 320 x 820 and 1440 x 1000 viewports. It does not inspect query workflows, API failures, focus order, keyboard behavior, loading states, citations, or backend connectivity.

## Required companion checks

Shadscan does not replace:

```bash
bun run typecheck
bun test
bun run api:lint
bun run api:test
bun run build
bun run doctor
bun run research:check
```

The primary browser path still needs manual or automated interaction coverage:

1. Configure a local provider.
2. Upload or load a document corpus.
3. Ask a grounded question.
4. Inspect citations, sources, trace data, and quality signals.
5. Exercise pending, empty, validation, failed-provider, abstention, and successful-answer states.
6. Confirm visible focus and keyboard access for configuration, document controls, query controls, drawers, and dialogs.

## Adoption rule

Do not add `--fail-under` to CI until a complete report has been reviewed and the remaining score is known to represent real JR AutoRAG requirements. Do not add a second UI system, duplicate components, or decorative features to satisfy a scanner rule.

## Current status

The integration scripts and workflow are present. The scan was not executed in the GitHub-only editing environment because the local shell could not resolve external hosts. No Shadscan score or pass is claimed.
