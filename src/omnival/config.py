"""Configuration objects matching the OmniVAL paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class MAValConfig:
    q_min: float = 0.85
    weights: Tuple[float, float, float, float] = (0.35, 0.35, 0.15, 0.15)
    filter_throughput_examples_per_sec: float = 48.0
    verifier_throughput_examples_per_sec: float = 11.0


@dataclass(frozen=True)
class PhaseAConfig:
    teacher: str = "Gemini 3.1 Pro"
    temperature: float = 0.7
    raw_trace_target: int = 120_000
    validated_trace_target: int = 110_000
    retention_rate: float = 0.917
    scaffolding_tools: Tuple[str, ...] = (
        "DB-ResNet",
        "SAM 2.1",
        "Whisper v3",
        "SigLIP",
        "pyannote",
        "TableFormer",
    )


@dataclass(frozen=True)
class StageB1Config:
    student: str = "Nemotron 3 Nano Omni"
    learning_rate: float = 1.5e-4
    effective_batch_size: int = 128
    sequence_length: int = 4096
    epochs: int = 3
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gpus: str = "8xH100 80GB"
    expected_wall_time_hours: float = 42.0


@dataclass(frozen=True)
class StageB2Config:
    learning_rate: float = 1.0e-5
    effective_batch_size: int = 64
    sequence_length: int = 4096
    epochs_per_iteration: int = 2
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_iterations: int = 20
    convergence_window: int = 3
    convergence_delta_min: float = 0.2
    convergence_delta_max: float = 0.4
    average_iterations: float = 13.4
    gpus: str = "8xH100 80GB"
    expected_wall_time_hours: float = 28.0


@dataclass(frozen=True)
class DataSplitConfig:
    train_fraction: float = 0.80
    refinement_fraction: float = 0.10
    test_fraction: float = 0.10
    train_target: int = 88_000
    refinement_target: int = 11_000
    test_target: int = 11_000


@dataclass(frozen=True)
class DeploymentConfig:
    active_params: str = "3B of 30B MoE"
    sequence_length: int = 4096
    inference_batch_size: int = 1
    inference_gpu: str = "1xA100 40GB"
    removes_scaffolding_at_inference: bool = True


@dataclass(frozen=True)
class OmniVALConfig:
    ma_val: MAValConfig = field(default_factory=MAValConfig)
    phase_a: PhaseAConfig = field(default_factory=PhaseAConfig)
    stage_b1: StageB1Config = field(default_factory=StageB1Config)
    stage_b2: StageB2Config = field(default_factory=StageB2Config)
    data_splits: DataSplitConfig = field(default_factory=DataSplitConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_config() -> OmniVALConfig:
    return OmniVALConfig()
