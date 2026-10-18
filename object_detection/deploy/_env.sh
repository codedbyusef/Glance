# Shared by the deploy scripts: loads object_detection/.env and checks the Vultr IP is set.
OD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$OD_DIR/.." && pwd)"

if [[ -f "$OD_DIR/.env" ]]; then
  set -a; source "$OD_DIR/.env"; set +a
fi

VULTR_HOST="${GLANCE_VULTR_HOST:-}"
SSH_USER="${GLANCE_SSH_USER:-root}"
SERVER_PORT="${GLANCE_SERVER_PORT:-8000}"
LOCAL_PORT="${GLANCE_LOCAL_PORT:-8000}"

if [[ -z "$VULTR_HOST" ]]; then
  echo "GLANCE_VULTR_HOST is not set." >&2
  echo "Copy object_detection/.env.example to object_detection/.env and put the Vultr IPv4 in it." >&2
  exit 1
fi

SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3)
if [[ -n "${GLANCE_SSH_KEY:-}" ]]; then
  SSH_OPTS+=(-i "${GLANCE_SSH_KEY/#\~/$HOME}")
fi
