"""LLM2: meeting documentation (summary, minutes, decisions, action items).

Consumes LLM1's refined_output.json. Reuses LLM1's provider/client infrastructure
(DeepSeek) but has its own prompt, schema, validation and outputs.

Public API: document_file(refined_json_path, outdir=None) -> (DocumentationResult, json_path, txt_path)
"""
from llm2.documenter import document_file, document_transcript  # noqa: F401
