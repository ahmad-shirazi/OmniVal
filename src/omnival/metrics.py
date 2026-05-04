"""Metric implementations used by MA-VAL and benchmark adapters."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .data import BoundingBox, CellSpan, Grounding, GroundingKind, SpeakerTurn, TimeSpan


def normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            replace_cost = previous[col_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls(predicted: str, target: str, threshold: float = 0.5) -> float:
    predicted_norm = normalize_text(predicted)
    target_norm = normalize_text(target)
    if not predicted_norm and not target_norm:
        return 1.0
    max_len = max(len(predicted_norm), len(target_norm), 1)
    normalized_distance = levenshtein_distance(predicted_norm, target_norm) / max_len
    return 1.0 - normalized_distance if normalized_distance < threshold else 0.0


def iou_2d(predicted: BoundingBox, target: BoundingBox) -> float:
    inter_x1 = max(predicted.x1, target.x1)
    inter_y1 = max(predicted.y1, target.y1)
    inter_x2 = min(predicted.x2, target.x2)
    inter_y2 = min(predicted.y2, target.y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    pred_area = max(0.0, predicted.x2 - predicted.x1) * max(0.0, predicted.y2 - predicted.y1)
    target_area = max(0.0, target.x2 - target.x1) * max(0.0, target.y2 - target.y1)
    union = pred_area + target_area - inter_area
    return inter_area / union if union > 0 else 0.0


def temporal_iou(predicted: TimeSpan, target: TimeSpan) -> float:
    intersection = max(0.0, min(predicted.end_ms, target.end_ms) - max(predicted.start_ms, target.start_ms))
    union = max(predicted.end_ms, target.end_ms) - min(predicted.start_ms, target.start_ms)
    return intersection / union if union > 0 else 0.0


def cell_f1(predicted: CellSpan, target: CellSpan) -> float:
    predicted_cells = set(predicted.cells())
    target_cells = set(target.cells())
    if not predicted_cells and not target_cells:
        return 1.0
    if not predicted_cells or not target_cells:
        return 0.0
    true_positive = len(predicted_cells & target_cells)
    precision = true_positive / len(predicted_cells)
    recall = true_positive / len(target_cells)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _speaker_at(turns: Sequence[SpeakerTurn], midpoint: float) -> str:
    for turn in turns:
        if turn.start_ms <= midpoint < turn.end_ms:
            return turn.speaker
    return "__none__"


def diarization_error_rate(predicted: Sequence[SpeakerTurn], target: Sequence[SpeakerTurn]) -> float:
    if not target:
        return 0.0 if not predicted else 1.0
    boundaries = set()
    for turn in list(predicted) + list(target):
        boundaries.add(turn.start_ms)
        boundaries.add(turn.end_ms)
    ordered = sorted(boundaries)
    if len(ordered) < 2:
        return 0.0
    total_ref = sum(max(0.0, turn.end_ms - turn.start_ms) for turn in target)
    error = 0.0
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        target_speaker = _speaker_at(target, midpoint)
        predicted_speaker = _speaker_at(predicted, midpoint)
        if target_speaker != "__none__" and predicted_speaker != target_speaker:
            error += end - start
        elif target_speaker == "__none__" and predicted_speaker != "__none__":
            error += end - start
    return error / total_ref if total_ref > 0 else 0.0


def overlap_for_grounding(predicted: Grounding, target: Grounding) -> float:
    if predicted.kind != target.kind:
        return 0.0
    if predicted.kind == GroundingKind.NONE:
        return 1.0
    if predicted.kind == GroundingKind.SPATIAL and isinstance(predicted.value, BoundingBox) and isinstance(target.value, BoundingBox):
        return iou_2d(predicted.value, target.value)
    if predicted.kind == GroundingKind.TEMPORAL and isinstance(predicted.value, TimeSpan) and isinstance(target.value, TimeSpan):
        return temporal_iou(predicted.value, target.value)
    if predicted.kind == GroundingKind.CELL and isinstance(predicted.value, CellSpan) and isinstance(target.value, CellSpan):
        return cell_f1(predicted.value, target.value)
    if predicted.kind == GroundingKind.DIARIZATION and isinstance(predicted.value, tuple) and isinstance(target.value, tuple):
        return max(0.0, 1.0 - diarization_error_rate(predicted.value, target.value))
    return 0.0


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
