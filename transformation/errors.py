from common.errors import PipelineError


class TransformationError(PipelineError):
    """Transformation model output was unusable (malformed/truncated/structurally invalid)."""
