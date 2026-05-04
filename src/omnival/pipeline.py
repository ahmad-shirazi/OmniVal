"""High-level OmniVAL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import OmniVALConfig, default_config
from .data import MultimodalExample
from .models import StudentModel, TeacherModel
from .scaffolding import ScaffoldingPipeline
from .training import PhaseAResult, PhaseATraceGenerator, SplitResult, StageB1Trainer, StageB2Refiner, StageB2Result, StratifiedSplitter
from .validator import MAValidator, QualityWeights


@dataclass
class OmniVALRunResult:
    phase_a: PhaseAResult
    splits: SplitResult
    stage_b1: dict[str, object]
    stage_b2: StageB2Result


class OmniVALPipeline:
    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        scaffolding: ScaffoldingPipeline | None = None,
        validator: MAValidator | None = None,
        config: OmniVALConfig | None = None,
    ) -> None:
        self.config = config or default_config()
        weights = QualityWeights.from_tuple(self.config.ma_val.weights)
        self.validator = validator or MAValidator(q_min=self.config.ma_val.q_min, weights=weights)
        self.teacher = teacher
        self.student = student
        self.scaffolding = scaffolding or ScaffoldingPipeline()

    def run(self, examples: Iterable[MultimodalExample]) -> OmniVALRunResult:
        phase_a = PhaseATraceGenerator(self.teacher, self.scaffolding, self.validator).run(examples)
        splits = StratifiedSplitter(self.config.data_splits).split(phase_a.records)
        stage_b1 = StageB1Trainer(self.student, self.config.stage_b1).run(splits.train)
        stage_b2 = StageB2Refiner(self.student, self.validator, self.config.stage_b2).run(splits.refinement)
        return OmniVALRunResult(phase_a=phase_a, splits=splits, stage_b1=stage_b1, stage_b2=stage_b2)
