"""Server wire-protocol and client URL tests with fake models (GLANCE_FAKE_MODELS=1).
No GPU, weights or Vultr instance needed. Run: pytest object_detection -v"""

import json

import numpy as np
import pytest

from object_detection import client
from object_detection import server
from object_detection.common import encode_jpeg


@pytest.fixture
def app_client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("GLANCE_FAKE_MODELS", "1")
    monkeypatch.delenv("GLANCE_TOKEN", raising=False)
    with TestClient(server.app) as tc:
        yield tc


def frame_bytes(frame_id: int) -> bytes:
    jpeg = encode_jpeg(np.zeros((360, 480, 3), dtype=np.uint8))
    return server.FRAME_HEADER.pack(frame_id, 123.0) + jpeg


def test_health(app_client):
    body = app_client.get("/health").json()
    assert body["status"] == "ok" and body["fake"] is True


def test_start_frame_cancel(app_client):
    with app_client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"cmd": "start", "target": "sheet of paper", "refer": ""}))
        started = ws.receive_json()
        assert started["type"] == "started"

        ws.send_bytes(frame_bytes(1))
        progress = ws.receive_json()
        assert progress["type"] == "progress" and progress["frame_id"] == 1
        assert progress["client_ts"] == 123.0 and progress["target_box"] is not None

        ws.send_text(json.dumps({"cmd": "cancel"}))
        end = ws.receive_json()
        assert end["type"] == "end" and end["status"] == "cancelled"
        assert end["session_id"] == started["session_id"]


def test_token_required_when_set(app_client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("GLANCE_TOKEN", "secret")
    with pytest.raises(WebSocketDisconnect):
        with app_client.websocket_connect("/ws") as ws:
            ws.receive_json()
    with app_client.websocket_connect("/ws?token=secret") as ws:
        ws.send_text(json.dumps({"cmd": "start", "target": "paper"}))
        assert ws.receive_json()["type"] == "started"


# --- client URL resolution -------------------------------------------------------

@pytest.fixture
def no_env(monkeypatch, tmp_path):
    for key in ("GLANCE_VISION_URL", "GLANCE_VULTR_HOST", "GLANCE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(client, "load_env_file", lambda *a, **k: None)


def _vultr_cfg(monkeypatch, **vultr):
    base = {"host": "", "server_port": 8000, "local_port": 8000, "use_tunnel": True, **vultr}
    monkeypatch.setattr(client, "load_config", lambda: {"vultr": base})


def test_url_tunnel_default(no_env, monkeypatch):
    _vultr_cfg(monkeypatch)
    assert client.server_url() == "ws://localhost:8000/ws"
    assert client.server_url("/health", scheme="http") == "http://localhost:8000/health"


def test_url_direct_needs_ip(no_env, monkeypatch):
    _vultr_cfg(monkeypatch, use_tunnel=False)
    with pytest.raises(client.ConfigError):
        client.server_url()
    monkeypatch.setenv("GLANCE_VULTR_HOST", "203.0.113.10")
    monkeypatch.setenv("GLANCE_TOKEN", "t")
    assert client.server_url() == "ws://203.0.113.10:8000/ws?token=t"


def test_url_override(no_env, monkeypatch):
    _vultr_cfg(monkeypatch)
    monkeypatch.setenv("GLANCE_VISION_URL", "wss://vision.example.com/ws")
    assert client.server_url() == "wss://vision.example.com/ws"
    assert client.server_url("/health", scheme="http") == "https://vision.example.com/health"
