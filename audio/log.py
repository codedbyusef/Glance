"""JSONL event log (CLAUDE.md section 13): one line per event in logs/audio-YYYYMMDD.jsonl."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("audio")


class EventLog:
    def __init__(self, log_dir: str | Path | None, echo=False, keep: bool = False):
        """echo: False, True (print each JSON line) or a callable that gets each event dict."""
        self.dir = Path(log_dir) if log_dir is not None else None
        self.echo = echo
        self.keep = keep or self.dir is None
        self.events: list[dict] = []  # kept in memory for tests and CLI summaries
        self._lock = threading.Lock()

    def write(self, event: str, **fields) -> None:
        rec = {"event": event, "mono": round(time.monotonic(), 4), "wall": round(time.time(), 4), **fields}
        line = json.dumps(rec, default=str)
        with self._lock:
            if self.keep:
                self.events.append(rec)
            if self.dir is not None:
                try:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    with open(self.dir / f"audio-{time.strftime('%Y%m%d')}.jsonl", "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except OSError:
                    logger.exception("event log write failed")
        if callable(self.echo):
            try:
                self.echo(rec)
            except Exception:
                logger.exception("event echo failed")
        elif self.echo:
            print(line, flush=True)

    def error(self, where: str, exc: BaseException | None = None, **fields) -> None:
        """Log an error event, plus a traceback through the logging module."""
        if exc is not None:
            logger.error("%s: %r", where, exc, exc_info=exc)
        self.write("error", where=where, error=repr(exc) if exc is not None else None, **fields)

    def of(self, event: str) -> list[dict]:
        return [e for e in self.events if e["event"] == event]


def null_log() -> EventLog:
    return EventLog(None)
