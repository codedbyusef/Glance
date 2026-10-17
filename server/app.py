"""FastAPI + WebSocket perception server, run on the Vultr GPU instance.

Processes only the newest frame per client and drops stale ones -- never queue,
a queue makes the glove lag the hand. Access via SSH tunnel, not a public port.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(websocket):
    raise NotImplementedError
