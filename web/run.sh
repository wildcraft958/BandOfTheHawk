#!/usr/bin/env bash
# Start the GAUNTLET web prototype with whatever the machine already has.
# Tries Docker, then a local Node build, then the committed single file over
# http, then tells you to open that file directly.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
URL="http://localhost:${PORT}"

open_browser() {
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  fi
}

case "${1:-}" in
  stop)
    docker compose down 2>/dev/null || true
    echo "stopped"
    exit 0
    ;;
  dev)
    npm install && exec npm run dev
    ;;
  build)
    npm ci && exec npm run build
    ;;
esac

# Tier 1: Docker. Assumes no toolchain on the host at all.
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "==> Docker Compose"
  docker compose up --build -d
  for _ in $(seq 1 90); do
    if curl -fsS "$URL" >/dev/null 2>&1; then break; fi
    sleep 1
  done
  echo "==> GAUNTLET web prototype: $URL      (stop with: ./run.sh stop)"
  open_browser
  exit 0
fi

# Tier 2: local Node build.
if command -v node >/dev/null 2>&1 && [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ]; then
  echo "==> local Node build (Docker not found)"
  npm ci && npm run build
  open_browser
  exec npx --yes vite preview --port "$PORT" --host
fi

# Tier 3: serve the committed build. No Node, no Docker.
if [ -f dist/index.html ] && command -v python3 >/dev/null 2>&1; then
  echo "==> serving the pre-built single file (no Node, no Docker)"
  open_browser
  exec python3 -m http.server "$PORT" --directory dist
fi

# Tier 4: nothing installed. The build is self-contained, so this still works.
echo "==> No Docker, Node, or Python found."
echo "    The build is fully self-contained. Open this file in a browser:"
echo "    $(pwd)/dist/index.html"
exit 0
