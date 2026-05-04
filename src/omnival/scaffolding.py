"""Training-time scaffolding adapters used by MA-VAL only."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Optional, Sequence

from .data import (
    ASRSegment,
    BoundingBox,
    CellSpan,
    KeyFrame,
    MultimodalExample,
    ScaffoldingEvidence,
    SpeakerTurn,
    TableCell,
    TextRegion,
    TimeSpan,
)


def _box(value: Any) -> Optional[BoundingBox]:
    if value is None or isinstance(value, BoundingBox):
        return value
    if isinstance(value, Mapping):
        return BoundingBox.from_dict(value)
    if isinstance(value, Sequence) and len(value) == 4:
        return BoundingBox(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    raise ValueError(f"Cannot parse bounding box from {value!r}")


def _span(value: Any) -> TimeSpan:
    if isinstance(value, TimeSpan):
        return value
    if isinstance(value, Mapping):
        return TimeSpan.from_dict(value)
    if isinstance(value, Sequence) and len(value) == 2:
        return TimeSpan(float(value[0]), float(value[1]))
    raise ValueError(f"Cannot parse time span from {value!r}")


def _cell(value: Any) -> CellSpan:
    if isinstance(value, CellSpan):
        return value
    if isinstance(value, Mapping):
        return CellSpan.from_dict(value)
    if isinstance(value, Sequence) and len(value) == 4:
        return CellSpan(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    raise ValueError(f"Cannot parse cell span from {value!r}")


class ScaffoldingTool(ABC):
    """Base class for paper scaffolding tools.

    Implementations read from example payloads in offline mode. Production
    subclasses can call DB-ResNet, SAM, Whisper, SigLIP, pyannote, or TableFormer.
    """

    name: str

    @abstractmethod
    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        raise NotImplementedError


class DBResNetTextDetector(ScaffoldingTool):
    name = "DB-ResNet"

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        regions = []
        for item in example.payload.get("text_regions", []):
            regions.append(TextRegion(text=str(item["text"]), box=_box(item.get("box")), source=self.name))
        return ScaffoldingEvidence(text_regions=regions, metadata={"tool": self.name})


class SAM21Segmenter(ScaffoldingTool):
    name = "SAM 2.1"

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        masks = {}
        for label, value in example.payload.get("masks", {}).items():
            parsed = _box(value)
            if parsed is not None:
                masks[str(label)] = parsed
        return ScaffoldingEvidence(masks=masks, metadata={"tool": self.name})


class WhisperV3ASR(ScaffoldingTool):
    name = "Whisper v3"

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        segments = []
        for item in example.payload.get("asr_segments", []):
            segments.append(ASRSegment(text=str(item["text"]), span=_span(item["span"]), speaker=item.get("speaker")))
        return ScaffoldingEvidence(asr_segments=segments, metadata={"tool": self.name})


class SigLIPKeyframeSelector(ScaffoldingTool):
    name = "SigLIP"

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        frames = []
        for item in example.payload.get("keyframes", []):
            frames.append(
                KeyFrame(
                    timestamp_ms=float(item["timestamp_ms"]),
                    description=str(item.get("description", "")),
                    box=_box(item.get("box")),
                )
            )
        return ScaffoldingEvidence(keyframes=frames, metadata={"tool": self.name})


class PyannoteDiarizer(ScaffoldingTool):
    name = "pyannote"

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        turns = []
        for item in example.payload.get("speaker_turns", []):
            turns.append(SpeakerTurn.from_dict(item))
        return ScaffoldingEvidence(speaker_turns=turns, metadata={"tool": self.name})


class TableFormerParser(ScaffoldingTool):
    name = "TableFormer"

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        cells = []
        for item in example.payload.get("table_cells", []):
            cells.append(TableCell(text=str(item["text"]), cell=_cell(item["cell"])))
        return ScaffoldingEvidence(table_cells=cells, metadata={"tool": self.name})


class ScaffoldingPipeline:
    def __init__(self, tools: Optional[Iterable[ScaffoldingTool]] = None) -> None:
        self.tools = list(
            tools
            if tools is not None
            else (
                DBResNetTextDetector(),
                SAM21Segmenter(),
                WhisperV3ASR(),
                SigLIPKeyframeSelector(),
                PyannoteDiarizer(),
                TableFormerParser(),
            )
        )

    def run(self, example: MultimodalExample) -> ScaffoldingEvidence:
        evidence = ScaffoldingEvidence(metadata={"scaffolding_used_for_training_only": True})
        for tool in self.tools:
            evidence.merge(tool.run(example))
        return evidence

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]
