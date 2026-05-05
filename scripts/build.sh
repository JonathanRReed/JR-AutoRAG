#!/usr/bin/env bash
set -eu

while IFS='=' read -r name _; do
  case "$name" in
    npm_*) unset "$name" ;;
  esac
done < <(env)

if [ -x /opt/homebrew/bin/bun ]; then
  BUN_BIN=/opt/homebrew/bin/bun
elif [ -x /usr/local/bin/bun ]; then
  BUN_BIN=/usr/local/bin/bun
else
  BUN_BIN=$(command -v bun)
fi

exec "$BUN_BIN" run ./build.ts "$@"
