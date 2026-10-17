#!/usr/bin/env bash
# Copy the code to the Vultr instance, build the image there and (re)start the server.
#   object-detection/deploy/deploy.sh            # real models
#   object-detection/deploy/deploy.sh --fake     # canned models, to check the wiring first
set -euo pipefail
source "$(dirname "$0")/_env.sh"

FAKE=0
[[ "${1:-}" == "--fake" ]] && FAKE=1
REMOTE_DIR="/opt/glance"
TARGET="$SSH_USER@$VULTR_HOST"

echo "== Checking the GPU and Docker on $VULTR_HOST"
ssh "${SSH_OPTS[@]}" "$TARGET" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && docker --version' || {
  echo "nvidia-smi or docker missing on the instance. Run deploy/setup_instance.sh first." >&2
  exit 1
}

echo "== Copying object-detection/ and hand-detection/ to $REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $REMOTE_DIR"
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.mp4' \
  "$REPO_DIR/object-detection" "$REPO_DIR/hand-detection" "$TARGET:$REMOTE_DIR/"

echo "== Building and starting the container (first build downloads the weights, a few minutes)"
ssh "${SSH_OPTS[@]}" "$TARGET" bash -s -- "$REMOTE_DIR" "$SERVER_PORT" "$FAKE" <<'REMOTE'
set -euo pipefail
cd "$1"
touch object-detection/.env
docker build -f object-detection/Dockerfile -t glance-vision .
docker rm -f glance-vision 2>/dev/null || true
# Bound to 127.0.0.1 so only the SSH tunnel reaches it. Drop the 127.0.0.1: part (and set
# GLANCE_TOKEN) only if the server has to be reachable without the tunnel.
docker run -d --gpus all --restart unless-stopped \
  -p "127.0.0.1:$2:8000" --env-file object-detection/.env \
  -e GLANCE_FAKE_MODELS="$( [[ "$3" == 1 ]] && echo 1 || echo 0 )" \
  --name glance-vision glance-vision
echo "Waiting for /health ..."
for i in $(seq 1 90); do
  if curl -fsS "http://localhost:$2/health"; then echo; exit 0; fi
  sleep 2
done
echo "Server did not come up; last logs:" >&2
docker logs --tail 50 glance-vision >&2
exit 1
REMOTE

echo "== Done. Next: object-detection/deploy/tunnel.sh, then python object-detection/client.py --health"
