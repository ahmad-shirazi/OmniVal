"""Teacher and student model interfaces and adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

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
    TraceRecord,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    role: str
    parameters: str = "unknown"
    open_source: bool = False
    notes: str = ""


MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("Gemini 3.1 Pro", "true native omni", "default teacher and closed-source baseline", open_source=False),
    ModelSpec("Gemini 2.5 Pro", "true native omni", "closed-source baseline", open_source=False),
    ModelSpec("Gemini 2.5 Flash", "true native omni", "teacher ablation", open_source=False),
    ModelSpec("GPT-4o", "true native omni", "closed-source baseline and teacher ablation", open_source=False),
    ModelSpec("Claude 4.5 Sonnet", "reasoning multimodal", "teacher ablation", open_source=False),
    ModelSpec("GPT-5", "reasoning multimodal", "teacher ablation", open_source=False),
    ModelSpec("Qwen3-VL-235B-A22B-Thinking", "vision-language MoE", "open-source teacher ablation", parameters="235B total / 22B active", open_source=True),
    ModelSpec("Llama 4-400B-A17B", "multimodal MoE", "open-source teacher ablation", parameters="400B total / 17B active", open_source=True),
    ModelSpec("Nemotron 3 Nano Omni", "true native omni MoE", "student and open-source baseline", parameters="30B total / 3B active", open_source=True),
    ModelSpec("Qwen3-Omni", "true native omni", "open-source baseline", open_source=True),
    ModelSpec("NExT-OMNI", "true native omni", "open-source baseline", open_source=True),
    ModelSpec("DocLayLLM", "document-specialist VLM", "document baseline", parameters="Llama3-7B", open_source=True),
    ModelSpec("LayoutLLM", "document-specialist VLM", "document baseline", parameters="Vicuna1.5-7B", open_source=True),
    ModelSpec("LayTextLLM", "document-specialist VLM", "document baseline", parameters="Llama2-7B", open_source=True),
    ModelSpec("DLaVA", "document-specialist VLM", "document baseline", parameters="Pixtral-12B", open_source=True),
)


class AdapterNotConfigured(RuntimeError):
    pass


class TeacherModel(ABC):
    name: str

    @abstractmethod
    def generate_trace(self, example: MultimodalExample, evidence: ScaffoldingEvidence) -> Trace:
        raise NotImplementedError


class ExternalTeacherAdapter(TeacherModel):
    def __init__(self, name: str, api_env_var: Optional[str] = None) -> None:
        self.name = name
        self.api_env_var = api_env_var

    def generate_trace(self, example: MultimodalExample, evidence: ScaffoldingEvidence) -> Trace:
        raise AdapterNotConfigured(
            f"{self.name} requires a project-specific API adapter. Use GroundTruthTemplateTeacher for offline smoke tests."
        )


class Gemini31ProTeacher(ExternalTeacherAdapter):
    def __init__(self) -> None:
        super().__init__("Gemini 3.1 Pro", api_env_var="GEMINI_API_KEY")


class Gemini25ProTeacher(ExternalTeacherAdapter):
    def __init__(self) -> None:
        super().__init__("Gemini 2.5 Pro", api_env_var="GEMINI_API_KEY")


class GPT4oTeacher(ExternalTeacherAdapter):
    def __init__(self) -> None:
        super().__init__("GPT-4o", api_env_var="OPENAI_API_KEY")


class GroundTruthTemplateTeacher(TeacherModel):
    """Offline teacher emulator for tests and pipeline dry runs.

    It uses the benchmark annotations the same way Phase A prompts condition the
    teacher on (M, q, answer_gt, grounding_gt). It is not a substitute for a real
    frontier teacher in experiments.
    """

    def __init__(self, name: str = "Gemini 3.1 Pro offline emulator") -> None:
        self.name = name

    def generate_trace(self, example: MultimodalExample, evidence: ScaffoldingEvidence) -> Trace:
        modality = example.grounding_gt.modality
        grounding_text = self._grounding_text(example.grounding_gt)
        reasoning = (
            f"Modality-routing: the answer is in the {modality.value} stream. "
            f"The question asks: {example.question} "
            f"I localize the relevant {example.grounding_gt.kind.value} evidence. {grounding_text} "
            f"Answer: {example.answer_gt}."
        )
        return Trace(
            reasoning=reasoning,
            answer=example.answer_gt,
            grounding=example.grounding_gt,
            modality_route=modality,
            raw={"teacher": self.name, "example_id": example.example_id, "evidence_text_count": len(evidence.all_text())},
        )

    def _grounding_text(self, grounding: Grounding) -> str:
        value = grounding.value
        if isinstance(value, BoundingBox):
            return f"Box: [{value.x1:.0f}, {value.y1:.0f}, {value.x2:.0f}, {value.y2:.0f}]."
        if isinstance(value, TimeSpan):
            return f"Time span: [{value.start_ms:.0f}, {value.end_ms:.0f}] ms."
        if isinstance(value, CellSpan):
            return f"Cell span: [{value.row_start}, {value.col_start}, {value.row_end}, {value.col_end}]."
        if isinstance(value, tuple):
            turns = ", ".join(f"{turn.speaker}:{turn.start_ms:.0f}-{turn.end_ms:.0f}ms" for turn in value)
            return f"Speaker turns: {turns}."
        return "No explicit coordinate required."


class TeacherFactory:
    @staticmethod
    def create(name: str, offline: bool = True) -> TeacherModel:
        if offline:
            return GroundTruthTemplateTeacher(name=f"{name} offline emulator")
        normalized = name.lower()
        if "gemini 3.1" in normalized:
            return Gemini31ProTeacher()
        if "gemini 2.5" in normalized:
            return Gemini25ProTeacher()
        if "gpt-4o" in normalized:
            return GPT4oTeacher()
        return ExternalTeacherAdapter(name)


class StudentModel(ABC):
    name: str

    @abstractmethod
    def predict(self, example: MultimodalExample) -> Trace:
        raise NotImplementedError

    @abstractmethod
    def fine_tune_supervised(self, records: Sequence[TraceRecord], **kwargs: object) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def fine_tune_corrections(self, records: Sequence[CorrectionExample], **kwargs: object) -> Dict[str, object]:
        raise NotImplementedError


class Nemotron3NanoOmniStudent(StudentModel):
    def __init__(self, backend_command: Optional[str] = None) -> None:
        self.name = "Nemotron 3 Nano Omni"
        self.backend_command = backend_command

    def predict(self, example: MultimodalExample) -> Trace:
        raise AdapterNotConfigured("Configure a local Nemotron inference backend before calling predict().")

    def fine_tune_supervised(self, records: Sequence[TraceRecord], **kwargs: object) -> Dict[str, object]:
        raise AdapterNotConfigured("Configure the 8xH100 full fine-tuning backend before Stage B1.")

    def fine_tune_corrections(self, records: Sequence[CorrectionExample], **kwargs: object) -> Dict[str, object]:
        raise AdapterNotConfigured("Configure the 8xH100 refinement backend before Stage B2.")


class MemorizingOmniStudent(StudentModel):
    """Small local student for smoke tests.

    It stores supervised and correction targets keyed by example_id, letting the
    end-to-end control flow run without large model dependencies.
    """

    def __init__(self, name: str = "Nemotron 3 Nano Omni offline student") -> None:
        self.name = name
        self._memory: Dict[str, Trace] = {}
        self.training_log: List[Dict[str, object]] = []

    def predict(self, example: MultimodalExample) -> Trace:
        if example.example_id in self._memory:
            return self._memory[example.example_id]
        return Trace(
            reasoning=f"Modality-routing: uncertain. I have not learned example {example.example_id} yet.",
            answer="",
            grounding=self._noisy_grounding(example.grounding_gt),
            modality_route=Modality.DOCUMENT if example.modality != Modality.DOCUMENT else Modality.AUDIO,
            raw={"student": self.name, "unseen": True},
        )

    def fine_tune_supervised(self, records: Sequence[TraceRecord], **kwargs: object) -> Dict[str, object]:
        for record in records:
            self._memory[record.example.example_id] = record.trace
        event = {"stage": "B1", "records": len(records), "kwargs": dict(kwargs)}
        self.training_log.append(event)
        return event

    def fine_tune_corrections(self, records: Sequence[CorrectionExample], **kwargs: object) -> Dict[str, object]:
        for record in records:
            self._memory[record.example.example_id] = record.target_trace
        event = {"stage": "B2", "records": len(records), "kwargs": dict(kwargs)}
        self.training_log.append(event)
        return event

    def _noisy_grounding(self, target: Grounding) -> Grounding:
        value = target.value
        if isinstance(value, BoundingBox):
            noisy = BoundingBox(value.x1 + 24, value.y1 + 18, value.x2 + 24, value.y2 + 18)
        elif isinstance(value, TimeSpan):
            noisy = TimeSpan(value.start_ms + 600, value.end_ms + 600)
        elif isinstance(value, CellSpan):
            noisy = CellSpan(value.row_start + 1, value.col_start, value.row_end + 1, value.col_end)
        elif isinstance(value, tuple):
            noisy = tuple(SpeakerTurn(turn.speaker, turn.start_ms + 250, turn.end_ms + 250) for turn in value)
        else:
            noisy = None
        return Grounding(kind=target.kind, modality=target.modality, value=noisy)


def model_catalog_by_name() -> Dict[str, ModelSpec]:
    return {spec.name: spec for spec in MODEL_CATALOG}
