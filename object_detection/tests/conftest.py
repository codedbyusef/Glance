import pytest


@pytest.fixture(autouse=True)
def no_local_env(monkeypatch):
    """Unit tests never read a developer's object_detection/.env (token, server URL) or the shell's: the
    server would demand a token the test client doesn't send, and the client would aim at a real server."""
    for module in ("object_detection.server", "object_detection.client"):
        monkeypatch.setattr(f"{module}.load_env_file", lambda *args, **kwargs: None)
    for name in ("GLANCE_TOKEN", "GLANCE_VISION_URL", "GLANCE_VULTR_HOST"):
        monkeypatch.delenv(name, raising=False)
