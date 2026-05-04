"""Benchmark catalog and lightweight evaluators for all paper benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .data import GroundingKind, Modality, TraceRecord
from .metrics import anls, mean, overlap_for_grounding
from .models import StudentModel


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    modality: Modality
    primary_metric: str
    higher_is_better: bool = True
    grounding_kind: GroundingKind = GroundingKind.NONE
    official_protocol: str = "Use official benchmark protocol."


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("DocVQA", Modality.DOCUMENT, "ANLS", True, GroundingKind.SPATIAL, "Official test ANLS; validation split mAP for hidden grounding."),
    BenchmarkSpec("VisualMRC", Modality.DOCUMENT, "ANLS", True, GroundingKind.SPATIAL),
    BenchmarkSpec("FUNSD", Modality.DOCUMENT, "ANLS", True, GroundingKind.SPATIAL),
    BenchmarkSpec("CORD", Modality.DOCUMENT, "ANLS", True, GroundingKind.SPATIAL),
    BenchmarkSpec("SROIE", Modality.DOCUMENT, "ANLS", True, GroundingKind.SPATIAL),
    BenchmarkSpec("OCRBenchV2-En", Modality.DOCUMENT, "official_score", True, GroundingKind.SPATIAL),
    BenchmarkSpec("MMLongBench-Doc", Modality.DOCUMENT, "official_score", True, GroundingKind.SPATIAL),
    BenchmarkSpec("CharXiv", Modality.CHART, "reasoning_score", True, GroundingKind.SPATIAL),
    BenchmarkSpec("ScreenSpot-Pro", Modality.GUI, "click_accuracy", True, GroundingKind.SPATIAL),
    BenchmarkSpec("OSWorld", Modality.GUI, "task_success", True, GroundingKind.SPATIAL),
    BenchmarkSpec("Video-MME", Modality.VIDEO, "official_score", True, GroundingKind.TEMPORAL),
    BenchmarkSpec("WorldSense", Modality.AUDIO_VISUAL, "official_score", True, GroundingKind.TEMPORAL),
    BenchmarkSpec("DailyOmni", Modality.AUDIO_VISUAL, "official_score", True, GroundingKind.TEMPORAL),
    BenchmarkSpec("VoiceBench", Modality.VOICE, "official_score", True, GroundingKind.TEMPORAL),
    BenchmarkSpec("HF Open ASR", Modality.ASR, "WER", False, GroundingKind.TEMPORAL, "Official 8-corpus aggregated WER."),
    BenchmarkSpec("DIHARD III", Modality.DIARIZATION, "DER", False, GroundingKind.DIARIZATION, "Track 1 oracle SAD; collar 0.0 and forgiveness collar 0.25s."),
)


class BenchmarkRegistry:
    def __init__(self, specs: Sequence[BenchmarkSpec] = BENCHMARKS) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> BenchmarkSpec:
        return self._specs[name]

    def all(self) -> List[BenchmarkSpec]:
        return list(self._specs.values())

    def names(self) -> List[str]:
        return list(self._specs)


class BenchmarkEvaluator:
    """Local evaluator for examples with public answers/grounding.

    Official benchmarks often need hidden servers or task-specific scripts. This
    class provides the common answer/grounding checks used by MA-VAL for local
    validation and dry runs.
    """

    def __init__(self, student: StudentModel) -> None:
        self.student = student

    def evaluate(self, records: Iterable[TraceRecord]) -> Dict[str, float]:
        answer_scores: List[float] = []
        grounding_scores: List[float] = []
        for record in records:
            prediction = self.student.predict(record.example)
            answer_scores.append(anls(prediction.answer, record.example.answer_gt))
            grounding_scores.append(overlap_for_grounding(prediction.grounding, record.example.grounding_gt))
        return {"ANLS_like": mean(answer_scores), "grounding_overlap": mean(grounding_scores)}
