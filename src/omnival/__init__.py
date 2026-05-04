"""OmniVAL reference implementation scaffold.

The package mirrors the paper's phases: validated teacher trace generation,
MA-VAL filtering/verifying, supervised CoT learning, iterative refinement, and
benchmark/result reporting. Heavy external systems are represented by explicit
adapters so the local code remains runnable without proprietary weights or APIs.
"""

from .config import OmniVALConfig, default_config
from .data import Grounding, GroundingKind, Modality, MultimodalExample, Trace
from .pipeline import OmniVALPipeline
from .validator import MAValidator, ValidationMode

__all__ = [
    "Grounding",
    "GroundingKind",
    "MAValidator",
    "Modality",
    "MultimodalExample",
    "OmniVALConfig",
    "OmniVALPipeline",
    "Trace",
    "ValidationMode",
    "default_config",
]
