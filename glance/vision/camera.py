"""Frame source: webcam, phone stream, or whatever the head-worn camera ends up feeding."""

from __future__ import annotations


class Camera:
    def __init__(self, source):
        self.source = source

    def read_frame(self):
        raise NotImplementedError
