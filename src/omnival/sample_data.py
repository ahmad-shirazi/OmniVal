"""Synthetic examples that exercise every modality and benchmark path."""

from __future__ import annotations

from typing import List

from .data import BoundingBox, Grounding, GroundingKind, Modality, MultimodalExample, SpeakerTurn, TimeSpan


def create_synthetic_pool() -> List[MultimodalExample]:
    examples: List[MultimodalExample] = []
    document_benchmarks = ["DocVQA", "VisualMRC", "FUNSD", "CORD", "SROIE", "OCRBenchV2-En", "MMLongBench-Doc"]
    for index, benchmark in enumerate(document_benchmarks, start=1):
        answer = f"total-{index}"
        box = BoundingBox(100 + index, 200 + index, 180 + index, 230 + index)
        examples.append(
            MultimodalExample(
                example_id=f"doc-{index}",
                modality=Modality.DOCUMENT,
                input_uri=f"synthetic://{benchmark}",
                question="What is the target total?",
                answer_gt=answer,
                grounding_gt=Grounding(GroundingKind.SPATIAL, Modality.DOCUMENT, box),
                benchmark=benchmark,
                payload={"text_regions": [{"text": answer, "box": box.to_dict()}]},
            )
        )
    chart_box = BoundingBox(315, 145, 412, 188)
    examples.append(
        MultimodalExample(
            example_id="chart-1",
            modality=Modality.CHART,
            input_uri="synthetic://CharXiv",
            question="Which method is highest at x=0.7?",
            answer_gt="OmniVAL",
            grounding_gt=Grounding(GroundingKind.SPATIAL, Modality.CHART, chart_box),
            benchmark="CharXiv",
            payload={"text_regions": [{"text": "OmniVAL", "box": chart_box.to_dict()}]},
        )
    )
    for index, benchmark in enumerate(("ScreenSpot-Pro", "OSWorld"), start=1):
        box = BoundingBox(140 + index, 32, 178 + index, 62)
        examples.append(
            MultimodalExample(
                example_id=f"gui-{index}",
                modality=Modality.GUI,
                input_uri=f"synthetic://{benchmark}",
                question="Click the save button.",
                answer_gt="click",
                grounding_gt=Grounding(GroundingKind.SPATIAL, Modality.GUI, box),
                benchmark=benchmark,
                payload={"text_regions": [{"text": "save", "box": box.to_dict()}], "masks": {"save_button": box.to_dict()}},
            )
        )
    video_span = TimeSpan.from_seconds(133.8, 136.1)
    examples.append(
        MultimodalExample(
            example_id="video-1",
            modality=Modality.VIDEO,
            input_uri="synthetic://Video-MME",
            question="What does the speaker hold up at the climax?",
            answer_gt="a red book",
            grounding_gt=Grounding(GroundingKind.TEMPORAL, Modality.VIDEO, video_span),
            benchmark="Video-MME",
            payload={"keyframes": [{"timestamp_ms": 134300, "description": "speaker holds up a red book"}]},
        )
    )
    for index, benchmark in enumerate(("WorldSense", "DailyOmni"), start=1):
        span = TimeSpan.from_seconds(44.6 + index, 46.3 + index)
        examples.append(
            MultimodalExample(
                example_id=f"av-{index}",
                modality=Modality.AUDIO_VISUAL,
                input_uri=f"synthetic://{benchmark}",
                question="At what time does the chef add salt?",
                answer_gt="adding salt",
                grounding_gt=Grounding(GroundingKind.TEMPORAL, Modality.AUDIO_VISUAL, span),
                benchmark=benchmark,
                payload={"asr_segments": [{"text": "now add a pinch of salt", "span": span.to_dict()}]},
            )
        )
    voice_span = TimeSpan.from_seconds(0.0, 3.2)
    examples.append(
        MultimodalExample(
            example_id="voice-1",
            modality=Modality.VOICE,
            input_uri="synthetic://VoiceBench",
            question="Follow the spoken instruction.",
            answer_gt="open settings",
            grounding_gt=Grounding(GroundingKind.TEMPORAL, Modality.VOICE, voice_span),
            benchmark="VoiceBench",
            payload={"asr_segments": [{"text": "open settings", "span": voice_span.to_dict()}]},
        )
    )
    asr_span = TimeSpan.from_seconds(0.0, 5.6)
    examples.append(
        MultimodalExample(
            example_id="asr-1",
            modality=Modality.ASR,
            input_uri="synthetic://HF Open ASR",
            question="Transcribe the audio.",
            answer_gt="welcome to the conference on multimodal learning",
            grounding_gt=Grounding(GroundingKind.TEMPORAL, Modality.ASR, asr_span),
            benchmark="HF Open ASR",
            payload={"asr_segments": [{"text": "welcome to the conference on multimodal learning", "span": asr_span.to_dict()}]},
        )
    )
    turns = (
        SpeakerTurn("A", 0.0, 4200.0),
        SpeakerTurn("B", 4200.0, 8600.0),
        SpeakerTurn("A", 8600.0, 12100.0),
    )
    examples.append(
        MultimodalExample(
            example_id="diar-1",
            modality=Modality.DIARIZATION,
            input_uri="synthetic://DIHARD III",
            question="Who spoke when?",
            answer_gt="A then B then A",
            grounding_gt=Grounding(GroundingKind.DIARIZATION, Modality.DIARIZATION, turns),
            benchmark="DIHARD III",
            payload={"speaker_turns": [turn.to_dict() for turn in turns]},
        )
    )
    return examples
