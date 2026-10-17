"""Open-vocabulary detector: find target by short noun phrase.

Candidates: google/owlv2-base-patch16-ensemble, IDEA-Research/grounding-dino-tiny.
OWLv2 boxes are relative to a padded square image -- convert coordinates in one
tested helper, don't duplicate the math.
"""

from __future__ import annotations


def detect(frame: bytes, target: str):
    raise NotImplementedError
