"""Transformation layer (Interpretation -> TRANSFORMATION -> Generation).

Turns LLM1's refined transcript into a lightweight, evidence-grounded structured representation
(topics, discussion units, decision/action candidates with status, relations). It writes prose NOTHING:
LLM2 (generation) consumes it as guidance but the refined transcript stays the source of truth.
"""
from transformation.transformer import transform_file, transform_transcript  # noqa: F401
