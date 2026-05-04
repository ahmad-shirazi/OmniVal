"""Phase A, Stage B1, and Stage B2 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import DataSplitConfig, StageB1Config, StageB2Config
from .data import CorrectionExample, Modality, MultimodalExample, TraceRecord
from .metrics import mean
from .models import StudentModel, TeacherModel
from .scaffolding import ScaffoldingPipeline
from .validator import MAValidator


@dataclass
class PhaseAResult:
    records: List[TraceRecord]
    rejected: int


@dataclass
class SplitResult:
    train: List[TraceRecord]
    refinement: List[TraceRecord]
    test: List[TraceRecord]


@dataclass
class StageB2Result:
    iterations: int
    score_history: List[float]
    correction_count: int


class PhaseATraceGenerator:
    def __init__(self, teacher: TeacherModel, scaffolding: ScaffoldingPipeline, validator: MAValidator) -> None:
        self.teacher = teacher
        self.scaffolding = scaffolding
        self.validator = validator

    def run(self, examples: Iterable[MultimodalExample]) -> PhaseAResult:
        records: List[TraceRecord] = []
        rejected = 0
        for example in examples:
            evidence = self.scaffolding.run(example)
            trace = self.teacher.generate_trace(example, evidence)
            feedback = self.validator.filter_teacher_trace(example, trace, evidence)
            if feedback.score.accepted:
                records.append(TraceRecord(example=example, trace=trace, scaffolding=evidence, quality=feedback.score.overall))
            else:
                rejected += 1
        return PhaseAResult(records=records, rejected=rejected)


class StratifiedSplitter:
    def __init__(self, config: DataSplitConfig) -> None:
        self.config = config

    def split(self, records: Sequence[TraceRecord]) -> SplitResult:
        by_modality: Dict[Modality, List[TraceRecord]] = {}
        for record in records:
            by_modality.setdefault(record.example.modality, []).append(record)
        train: List[TraceRecord] = []
        refinement: List[TraceRecord] = []
        test: List[TraceRecord] = []
        for group in by_modality.values():
            group = sorted(group, key=lambda item: item.example.example_id)
            n = len(group)
            if n == 1:
                train.extend(group)
                continue
            if n == 2:
                train.append(group[0])
                refinement.append(group[1])
                continue
            train_count = max(1, min(int(round(n * self.config.train_fraction)), n - 2))
            refinement_count = max(1, min(int(round(n * self.config.refinement_fraction)), n - train_count - 1))
            train.extend(group[:train_count])
            refinement.extend(group[train_count : train_count + refinement_count])
            test.extend(group[train_count + refinement_count :])
        return SplitResult(train=train, refinement=refinement, test=test)


class StageB1Trainer:
    def __init__(self, student: StudentModel, config: StageB1Config) -> None:
        self.student = student
        self.config = config

    def run(self, records: Sequence[TraceRecord]) -> Dict[str, object]:
        return self.student.fine_tune_supervised(
            records,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.effective_batch_size,
            epochs=self.config.epochs,
            sequence_length=self.config.sequence_length,
            optimizer=self.config.optimizer,
        )


class ConvergenceTracker:
    def __init__(self, window: int, delta_min: float, delta_max: float) -> None:
        self.window = window
        self.delta_min = delta_min
        self.delta_max = delta_max

    def converged(self, scores: Sequence[float]) -> bool:
        if len(scores) < self.window + 1:
            return False
        deltas = [abs(scores[index] - scores[index - 1]) for index in range(len(scores) - self.window, len(scores))]
        return mean(deltas) < self.delta_min and max(deltas) < self.delta_max


class StageB2Refiner:
    def __init__(self, student: StudentModel, validator: MAValidator, config: StageB2Config) -> None:
        self.student = student
        self.validator = validator
        self.config = config
        self.convergence = ConvergenceTracker(
            window=config.convergence_window,
            delta_min=config.convergence_delta_min,
            delta_max=config.convergence_delta_max,
        )

    def run(self, records: Sequence[TraceRecord]) -> StageB2Result:
        if not records:
            return StageB2Result(iterations=0, score_history=[], correction_count=0)
        score_history: List[float] = []
        total_corrections = 0
        iterations = 0
        for iteration in range(1, self.config.max_iterations + 1):
            iterations = iteration
            corrections: List[CorrectionExample] = []
            for record in records:
                prediction = self.student.predict(record.example)
                feedback = self.validator.verify_student_prediction(record.example, prediction, record.scaffolding)
                if not feedback.score.accepted:
                    corrections.append(
                        CorrectionExample(
                            example=record.example,
                            feedback=feedback.as_instruction(),
                            target_trace=feedback.corrected_trace or record.trace,
                            quality=feedback.score.overall,
                        )
                    )
            if corrections:
                self.student.fine_tune_corrections(
                    corrections,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.effective_batch_size,
                    epochs=self.config.epochs_per_iteration,
                    iteration=iteration,
                )
                total_corrections += len(corrections)
            post_scores = []
            for record in records:
                prediction = self.student.predict(record.example)
                feedback = self.validator.verify_student_prediction(record.example, prediction, record.scaffolding)
                post_scores.append(feedback.score.overall)
            score_history.append(mean(post_scores))
            if self.convergence.converged(score_history):
                break
        return StageB2Result(iterations=iterations, score_history=score_history, correction_count=total_corrections)
