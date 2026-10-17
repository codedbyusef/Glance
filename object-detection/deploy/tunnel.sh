#!/usr/bin/env bash
# Forward laptop localhost:8000 to the vision server on the Vultr instance.
# Leave it running; the client then connects to ws://localhost:8000/ws.
#   object-detection/deploy/tunnel.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "Tunnel: localhost:$LOCAL_PORT -> $SSH_USER@$VULTR_HOST:$SERVER_PORT (Ctrl-C to stop)"
exec ssh -N "${SSH_OPTS[@]}" -o ExitOnForwardFailure=yes \
  -L "$LOCAL_PORT:localhost:$SERVER_PORT" "$SSH_USER@$VULTR_HOST"
