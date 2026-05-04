"""MA-VAL: modality-aware rule-based validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .data import (
    BoundingBox,
    CellSpan,
    CorrectionExample,
    Grounding,
    GroundingKind,
    Modality,
    MultimodalExample,
    ScaffoldingEvidence,
    SpeakerTurn,
    TimeSpan,
    Trace,
)
from .metrics import anls, overlap_for_grounding


class ValidationMode(str, Enum):
    FILTER = "filter"
    VERIFIER = "verifier"


@dataclass(frozen=True)
class QualityWeights:
    answer: float = 0.35
    grounding: float = 0.35
    modality: float = 0.15
    reasoning: float = 0.15

    @classmethod
    def from_tuple(cls, value: Tuple[float, float, float, float]) -> "QualityWeights":
        return cls(answer=value[0], grounding=value[1], modality=value[2], reasoning=value[3])


@dataclass(frozen=True)
class ValidationScore:
    q_answer: float
    q_grounding: float
    q_modality: float
    q_reasoning: float
    overall: float
    accepted: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "q_answer": self.q_answer,
            "q_grounding": self.q_grounding,
            "q_modality": self.q_modality,
            "q_reasoning": self.q_reasoning,
            "overall": self.overall,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class CorrectionVector:
    kind: GroundingKind
    values: Tuple[float, ...]

    def describe(self) -> str:
        if self.kind == GroundingKind.SPATIAL:
            return f"pixel deltas dx1={self.values[0]:.1f}, dy1={self.values[1]:.1f}, dx2={self.values[2]:.1f}, dy2={self.values[3]:.1f}"
        if self.kind == GroundingKind.TEMPORAL:
            return f"millisecond deltas dt1={self.values[0]:.1f}, dt2={self.values[1]:.1f}"
        if self.kind == GroundingKind.CELL:
            return f"cell deltas dr1={self.values[0]:.0f}, dc1={self.values[1]:.0f}, dr2={self.values[2]:.0f}, dc2={self.values[3]:.0f}"
        return "diarization turn corrections"


@dataclass
class ValidationFeedback:
    score: ValidationScore
    messages: List[str] = field(default_factory=list)
    correction_vector: Optional[CorrectionVector] = None
    corrected_trace: Optional[Trace] = None
    priority_fixes: List[str] = field(default_factory=list)

    def as_instruction(self) -> str:
        parts = list(self.messages)
        if self.correction_vector is not None:
            parts.append(f"Apply {self.correction_vector.describe()}.")
        if self.priority_fixes:
            parts.append("Priority fixes: " + "; ".join(self.priority_fixes))
        return " ".join(parts)


class MultimodalGroundingEngine:
    def overlap(self, predicted: Grounding, target: Grounding) -> float:
        return overlap_for_grounding(predicted, target)

    def correction_delta(self, predicted: Grounding, target: Grounding) -> Optional[CorrectionVector]:
        if predicted.kind != target.kind:
            return CorrectionVector(target.kind, tuple())
        if isinstance(predicted.value, BoundingBox) and isinstance(target.value, BoundingBox):
            return CorrectionVector(GroundingKind.SPATIAL, predicted.value.delta_to(target.value))
        if isinstance(predicted.value, TimeSpan) and isinstance(target.value, TimeSpan):
            return CorrectionVector(GroundingKind.TEMPORAL, predicted.value.delta_to(target.value))
        if isinstance(predicted.value, CellSpan) and isinstance(target.value, CellSpan):
            return CorrectionVector(GroundingKind.CELL, tuple(float(value) for value in predicted.value.delta_to(target.value)))
        if isinstance(predicted.value, tuple) and isinstance(target.value, tuple):
            return CorrectionVector(GroundingKind.DIARIZATION, tuple())
        return None


class AnswerValidator:
    def score(self, predicted: str, target: str, evidence: ScaffoldingEvidence) -> float:
        text_score = anls(predicted, target)
        presence = 1.0 if evidence.contains_text(predicted) or evidence.contains_text(target) or anls(predicted, target) == 1.0 else 0.0
        return 0.7 * text_score + 0.3 * presence


class ModalityAwareGroundingValidator:
    def __init__(self, engine: Optional[MultimodalGroundingEngine] = None) -> None:
        self.engine = engine or MultimodalGroundingEngine()

    def score(self, predicted: Grounding, target: Grounding) -> float:
        overlap = self.engine.overlap(predicted, target)
        same_region = 1.0 if overlap >= 0.5 else 0.0
        return 0.8 * overlap + 0.2 * same_region


class CrossModalReasoningValidator:
    def score(self, trace: Trace, target_grounding: Grounding, evidence: ScaffoldingEvidence) -> float:
        modality_match = 1.0 if trace.grounding.modality == target_grounding.modality else 0.0
        text = f"{trace.modality_route.value} {trace.reasoning}".lower()
        modality_name = target_grounding.modality.value.replace("_", " ")
        route_match = 1.0 if modality_name in text or target_grounding.modality.value in text else 0.0
        scaffolding_consistent = 1.0 if evidence.contains_text(trace.answer) else 0.0
        if trace.grounding.kind == target_grounding.kind and overlap_for_grounding(trace.grounding, target_grounding) >= 0.3:
            scaffolding_consistent = 1.0
        return 0.5 * modality_match + 0.3 * route_match + 0.2 * scaffolding_consistent


class ReasoningFeedbackGenerator:
    _kind_terms = {
        GroundingKind.SPATIAL: ("box", "bbox", "pixel", "click", "region"),
        GroundingKind.TEMPORAL: ("span", "time", "timestamp", "moment", "audio"),
        GroundingKind.CELL: ("cell", "row", "column", "table"),
        GroundingKind.DIARIZATION: ("speaker", "turn", "diarization"),
        GroundingKind.NONE: ("answer",),
    }

    def score(self, trace: Trace) -> float:
        structural = 1.0 if trace.reasoning.strip() and trace.answer.strip() and trace.grounding.kind else 0.0
        coordinate = 1.0 if self._has_valid_coordinate(trace.grounding) else 0.0
        text = trace.reasoning.lower()
        terms = self._kind_terms.get(trace.grounding.kind, ())
        aligned = 1.0 if any(term in text for term in terms) else 0.0
        return (structural + coordinate + aligned) / 3.0

    def feedback(self, trace: Trace, example: MultimodalExample, score: ValidationScore, correction: Optional[CorrectionVector]) -> ValidationFeedback:
        messages: List[str] = []
        priorities: List[str] = []
        if score.q_answer < 0.85:
            messages.append(f"Answer error: predicted '{trace.answer}', ground truth is '{example.answer_gt}'.")
            priorities.append("repair answer text")
        if score.q_grounding < 0.85:
            messages.append("Grounding error: predicted grounding does not sufficiently overlap the target.")
            priorities.append("re-localize grounding")
        if score.q_modality < 0.85:
            messages.append(f"Modality routing error: predicted {trace.grounding.modality.value}, expected {example.grounding_gt.modality.value}.")
            priorities.append("fix modality route")
        if score.q_reasoning < 0.85:
            messages.append("Reasoning issue: include modality route, evidence lookup, answer, and coordinate justification.")
            priorities.append("complete reasoning structure")
        if not messages:
            messages.append("Prediction is valid under MA-VAL.")
        target_trace = Trace(
            reasoning=self._target_reasoning(example),
            answer=example.answer_gt,
            grounding=example.grounding_gt,
            modality_route=example.grounding_gt.modality,
            raw={"source": "ma_val_corrected_target", "example_id": example.example_id},
        )
        return ValidationFeedback(score=score, messages=messages, correction_vector=correction, corrected_trace=target_trace, priority_fixes=priorities)

    def _has_valid_coordinate(self, grounding: Grounding) -> bool:
        if grounding.kind == GroundingKind.NONE:
            return True
        return grounding.value is not None

    def _target_reasoning(self, example: MultimodalExample) -> str:
        kind = example.grounding_gt.kind.value
        modality = example.grounding_gt.modality.value
        return f"Modality-routing: the answer is in the {modality} stream. Locate the {kind} grounding, verify it against the question evidence, then emit the answer and grounding."


class MAValidator:
    """Five-module MA-VAL implementation with Filter and Verifier modes."""

    def __init__(self, q_min: float = 0.85, weights: Optional[QualityWeights] = None) -> None:
        self.q_min = q_min
        self.weights = weights or QualityWeights()
        self.grounding_engine = MultimodalGroundingEngine()
        self.answer_validator = AnswerValidator()
        self.grounding_validator = ModalityAwareGroundingValidator(self.grounding_engine)
        self.modality_validator = CrossModalReasoningValidator()
        self.reasoning_generator = ReasoningFeedbackGenerator()

    def evaluate(self, example: MultimodalExample, trace: Trace, evidence: ScaffoldingEvidence, mode: ValidationMode) -> ValidationFeedback:
        q_answer = self.answer_validator.score(trace.answer, example.answer_gt, evidence)
        q_grounding = self.grounding_validator.score(trace.grounding, example.grounding_gt)
        q_modality = self.modality_validator.score(trace, example.grounding_gt, evidence)
        q_reasoning = self.reasoning_generator.score(trace)
        overall = (
            self.weights.answer * q_answer
            + self.weights.grounding * q_grounding
            + self.weights.modality * q_modality
            + self.weights.reasoning * q_reasoning
        )
        score = ValidationScore(
            q_answer=q_answer,
            q_grounding=q_grounding,
            q_modality=q_modality,
            q_reasoning=q_reasoning,
            overall=overall,
            accepted=overall > self.q_min,
        )
        correction = self.grounding_engine.correction_delta(trace.grounding, example.grounding_gt)
        if mode == ValidationMode.FILTER:
            message = "ACCEPT" if score.accepted else "REJECT"
            return ValidationFeedback(score=score, messages=[f"MA-VAL Filter: Q={overall:.3f} -> {message}"])
        return self.reasoning_generator.feedback(trace, example, score, correction)

    def filter_teacher_trace(self, example: MultimodalExample, trace: Trace, evidence: ScaffoldingEvidence) -> ValidationFeedback:
        return self.evaluate(example, trace, evidence, ValidationMode.FILTER)

    def verify_student_prediction(self, example: MultimodalExample, trace: Trace, evidence: ScaffoldingEvidence) -> ValidationFeedback:
        return self.evaluate(example, trace, evidence, ValidationMode.VERIFIER)

    def correction_example(self, example: MultimodalExample, trace: Trace, evidence: ScaffoldingEvidence) -> CorrectionExample:
        feedback = self.verify_student_prediction(example, trace, evidence)
        target = feedback.corrected_trace or trace
        return CorrectionExample(example=example, feedback=feedback.as_instruction(), target_trace=target, quality=feedback.score.overall)
