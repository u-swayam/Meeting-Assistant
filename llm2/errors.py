from common.errors import PipelineError


class LLM2OutputError(PipelineError):
    """The model's output was unusable (malformed/truncated/structurally invalid)."""
