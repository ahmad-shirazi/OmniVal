"""Core data structures for grounded omni-modal reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class Modality(str, Enum):
    DOCUMENT = "document"
    CHART = "chart"
    GUI = "gui"
    VIDEO = "video"
    AUDIO = "audio"
    AUDIO_VISUAL = "audio_visual"
    VOICE = "voice"
    ASR = "asr"
    DIARIZATION = "diarization"
    TABLE = "table"


class GroundingKind(str, Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CELL = "cell"
    DIARIZATION = "diarization"
    NONE = "none"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid box coordinates: {self.as_tuple()}")

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def delta_to(self, target: "BoundingBox") -> Tuple[float, float, float, float]:
        return (
            target.x1 - self.x1,
            target.y1 - self.y1,
            target.x2 - self.x2,
            target.y2 - self.y2,
        )

    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundingBox":
        return cls(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))


@dataclass(frozen=True)
class TimeSpan:
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(f"Invalid time span: {self.as_tuple()}")

    @classmethod
    def from_seconds(cls, start_s: float, end_s: float) -> "TimeSpan":
        return cls(start_s * 1000.0, end_s * 1000.0)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.start_ms, self.end_ms)

    def delta_to(self, target: "TimeSpan") -> Tuple[float, float]:
        return (target.start_ms - self.start_ms, target.end_ms - self.end_ms)

    def to_dict(self) -> Dict[str, float]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimeSpan":
        return cls(float(value["start_ms"]), float(value["end_ms"]))


@dataclass(frozen=True)
class CellSpan:
    row_start: int
    col_start: int
    row_end: int
    col_end: int

    def __post_init__(self) -> None:
        if self.row_end < self.row_start or self.col_end < self.col_start:
            raise ValueError(f"Invalid cell span: {self.as_tuple()}")

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.row_start, self.col_start, self.row_end, self.col_end)

    def cells(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (row, col)
            for row in range(self.row_start, self.row_end + 1)
            for col in range(self.col_start, self.col_end + 1)
        )

    def delta_to(self, target: "CellSpan") -> Tuple[int, int, int, int]:
        return (
            target.row_start - self.row_start,
            target.col_start - self.col_start,
            target.row_end - self.row_end,
            target.col_end - self.col_end,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "row_start": self.row_start,
            "col_start": self.col_start,
            "row_end": self.row_end,
            "col_end": self.col_end,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CellSpan":
        return cls(
            int(value["row_start"]),
            int(value["col_start"]),
            int(value["row_end"]),
            int(value["col_end"]),
        )


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(f"Invalid speaker turn: {self.speaker} {self.start_ms}-{self.end_ms}")

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {"speaker": self.speaker, "start_ms": self.start_ms, "end_ms": self.end_ms}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpeakerTurn":
        return cls(str(value["speaker"]), float(value["start_ms"]), float(value["end_ms"]))


GroundingValue = Union[BoundingBox, TimeSpan, CellSpan, Tuple[SpeakerTurn, ...], None]


@dataclass(frozen=True)
class Grounding:
    kind: GroundingKind
    modality: Modality
    value: GroundingValue = None

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.value, BoundingBox):
            payload: Any = self.value.to_dict()
        elif isinstance(self.value, TimeSpan):
            payload = self.value.to_dict()
        elif isinstance(self.value, CellSpan):
            payload = self.value.to_dict()
        elif isinstance(self.value, tuple):
            payload = [turn.to_dict() for turn in self.value]
        else:
            payload = None
        return {"kind": self.kind.value, "modality": self.modality.value, "value": payload}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Grounding":
        kind = GroundingKind(value["kind"])
        modality = Modality(value["modality"])
        payload = value.get("value")
        if kind == GroundingKind.SPATIAL and payload is not None:
            parsed: GroundingValue = BoundingBox.from_dict(payload)
        elif kind == GroundingKind.TEMPORAL and payload is not None:
            parsed = TimeSpan.from_dict(payload)
        elif kind == GroundingKind.CELL and payload is not None:
            parsed = CellSpan.from_dict(payload)
        elif kind == GroundingKind.DIARIZATION and payload is not None:
            parsed = tuple(SpeakerTurn.from_dict(item) for item in payload)
        else:
            parsed = None
        return cls(kind=kind, modality=modality, value=parsed)


@dataclass(frozen=True)
class TextRegion:
    text: str
    box: Optional[BoundingBox] = None
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "box": None if self.box is None else self.box.to_dict(), "source": self.source}


@dataclass(frozen=True)
class ASRSegment:
    text: str
    span: TimeSpan
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "span": self.span.to_dict(), "speaker": self.speaker}


@dataclass(frozen=True)
class TableCell:
    text: str
    cell: CellSpan

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "cell": self.cell.to_dict()}


@dataclass(frozen=True)
class KeyFrame:
    timestamp_ms: float
    description: str = ""
    box: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "description": self.description,
            "box": None if self.box is None else self.box.to_dict(),
        }


@dataclass
class ScaffoldingEvidence:
    text_regions: List[TextRegion] = field(default_factory=list)
    asr_segments: List[ASRSegment] = field(default_factory=list)
    table_cells: List[TableCell] = field(default_factory=list)
    speaker_turns: List[SpeakerTurn] = field(default_factory=list)
    keyframes: List[KeyFrame] = field(default_factory=list)
    masks: Dict[str, BoundingBox] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def all_text(self) -> List[str]:
        values = [region.text for region in self.text_regions]
        values.extend(segment.text for segment in self.asr_segments)
        values.extend(cell.text for cell in self.table_cells)
        values.extend(frame.description for frame in self.keyframes if frame.description)
        return values

    def contains_text(self, text: str) -> bool:
        needle = " ".join(text.lower().strip().split())
        if not needle:
            return False
        return any(needle in " ".join(value.lower().split()) for value in self.all_text())

    def merge(self, other: "ScaffoldingEvidence") -> "ScaffoldingEvidence":
        self.text_regions.extend(other.text_regions)
        self.asr_segments.extend(other.asr_segments)
        self.table_cells.extend(other.table_cells)
        self.speaker_turns.extend(other.speaker_turns)
        self.keyframes.extend(other.keyframes)
        self.masks.update(other.masks)
        self.metadata.update(other.metadata)
        return self


@dataclass
class MultimodalExample:
    example_id: str
    modality: Modality
    input_uri: str
    question: str
    answer_gt: str
    grounding_gt: Grounding
    benchmark: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "modality": self.modality.value,
            "input_uri": self.input_uri,
            "question": self.question,
            "answer_gt": self.answer_gt,
            "grounding_gt": self.grounding_gt.to_dict(),
            "benchmark": self.benchmark,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultimodalExample":
        return cls(
            example_id=str(value["example_id"]),
            modality=Modality(value["modality"]),
            input_uri=str(value["input_uri"]),
            question=str(value["question"]),
            answer_gt=str(value["answer_gt"]),
            grounding_gt=Grounding.from_dict(value["grounding_gt"]),
            benchmark=value.get("benchmark"),
            payload=dict(value.get("payload", {})),
        )


@dataclass
class Trace:
    reasoning: str
    answer: str
    grounding: Grounding
    modality_route: Modality
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "answer": self.answer,
            "grounding": self.grounding.to_dict(),
            "modality_route": self.modality_route.value,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Trace":
        return cls(
            reasoning=str(value["reasoning"]),
            answer=str(value["answer"]),
            grounding=Grounding.from_dict(value["grounding"]),
            modality_route=Modality(value["modality_route"]),
            raw=dict(value.get("raw", {})),
        )


@dataclass
class TraceRecord:
    example: MultimodalExample
    trace: Trace
    scaffolding: ScaffoldingEvidence
    quality: float


@dataclass
class CorrectionExample:
    example: MultimodalExample
    feedback: str
    target_trace: Trace
    quality: float


def modality_counts(examples: Iterable[MultimodalExample]) -> Dict[Modality, int]:
    counts: Dict[Modality, int] = {}
    for example in examples:
        counts[example.modality] = counts.get(example.modality, 0) + 1
    return counts
