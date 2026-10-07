"""LLM1: domain-aware transcript refinement (self-contained module).

Public API:  refine_file(structured_json_path, outdir=None) -> RefinementResult
"""
from llm1.refiner import refine_file, refine_transcript  # noqa: F401
