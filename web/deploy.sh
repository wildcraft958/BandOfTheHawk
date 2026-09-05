#!/usr/bin/env bash
# Publish the built prototype to a public URL.
#
# Both providers upload this folder directly rather than reading the git repo,
# which is what we need: web/ is gitignored, so a git-connected deploy would
# find nothing.
#
#   ./deploy.sh vercel-now   no account needed, returns a claimable URL
#   ./deploy.sh login        authenticate, then use ./deploy.sh vercel
#   ./deploy.sh vercel       deploys to your account
#   ./deploy.sh netlify
set -euo pipefail
cd "$(dirname "$0")"

PROVIDER="${1:-vercel}"

# The submission URL. Locked: it is the production alias of the Vercel project
# named below, and every update has to land on it. Do not change either value.
LOCKED_URL="https://gauntlet-eight-theta.vercel.app/"
LOCKED_PROJECT="gauntlet"

echo "==> building"
npm run build

if [ ! -f dist/index.html ]; then
  echo "build produced no dist/index.html" >&2
  exit 1
fi

# Vercel names the project after the directory it uploads, so uploading "dist"
# produces project "dist" and a URL that reads dist-<hash>. Stage the build in a
# named directory instead.
STAGE=".deploy/gauntlet"
rm -rf .deploy
mkdir -p "$STAGE"
cp dist/index.html "$STAGE/index.html"
[ -f vercel.json ] && cp vercel.json "$STAGE/vercel.json"

# Carry the project link into the staging directory. Without it Vercel picks the
# project by directory name, which is how a stray project called "dist" and a
# second URL got created once already. With it, the deploy targets the project by
# id and the locked URL cannot move.
if [ ! -f .vercel/project.json ]; then
  echo "no .vercel/project.json. Run: npx vercel link --yes --project $LOCKED_PROJECT" >&2
  exit 1
fi
LINKED=$(python3 -c "import json;print(json.load(open('.vercel/project.json'))['projectName'])")
if [ "$LINKED" != "$LOCKED_PROJECT" ]; then
  echo "linked to project '$LINKED' but the submission URL belongs to '$LOCKED_PROJECT'." >&2
  echo "refusing to deploy: this would publish to a different URL." >&2
  exit 1
fi
mkdir -p "$STAGE/.vercel"
cp .vercel/project.json "$STAGE/.vercel/project.json"

case "$PROVIDER" in
  vercel)
    echo "==> deploying dist/ to Vercel"
    npx --yes vercel@latest deploy "$STAGE" --prod --yes
    ;;
  vercel-now)
    # No account needed. Vercel returns a claimable URL that you can attach to
    # an account later, which is the fastest way to a public link.
    echo "==> deploying dist/ to Vercel without logging in"
    npx --yes vercel@latest deploy "$STAGE" --temporary --yes
    ;;
  login)
    exec npx --yes vercel@latest login
    ;;
  netlify)
    echo "==> deploying dist/ to Netlify"
    npx --yes netlify-cli@latest deploy --dir="$STAGE" --site-name=gauntlet-defense-lab --prod
    ;;
  *)
    echo "unknown target: $PROVIDER (use vercel-now, login, vercel, or netlify)" >&2
    exit 1
    ;;
esac

# A deploy that reports success but leaves the locked URL serving the old build
# is the failure worth catching, so check the bytes rather than the exit code.
if [ "$PROVIDER" = "vercel" ]; then
  echo "==> checking $LOCKED_URL serves this build"
  want=$(md5 -q dist/index.html)
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    got=$(curl -sS -L "$LOCKED_URL" | md5 -q)
    if [ "$want" = "$got" ]; then
      echo "    ok, $LOCKED_URL is serving $want"
      exit 0
    fi
    sleep 3
  done
  echo "    $LOCKED_URL still serves $got, expected $want" >&2
  exit 1
fi
