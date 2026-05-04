"""Paper-reported baselines, main results, and ablations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


PAPER_RESULTS: Dict[str, Any] = {
    "table_1_document_text_grounding": {
        "metrics": ["DocVQA_ANLS", "DocVQA_mAP", "VisualMRC_ANLS", "VisualMRC_mAP", "FUNSD_ANLS", "FUNSD_mAP", "CORD_ANLS", "CORD_mAP", "SROIE_ANLS", "SROIE_mAP"],
        "rows": {
            "DocLayLLM": [78.4, None, 55.0, None, 84.1, None, 71.3, None, 84.3, None],
            "LayoutLLM": [74.3, None, 55.8, None, 80.0, None, 63.1, None, 72.1, None],
            "LayTextLLM": [77.2, None, 41.7, None, 81.0, None, 82.5, None, 96.1, None],
            "DLaVA": [85.9, 46.2, 52.1, None, 87.6, 45.5, 84.4, 57.9, 91.4, None],
            "Gemini 3.1 Pro": [94.6, 86.9, 77.4, 72.5, 94.8, 85.7, 91.6, 82.4, 96.8, 83.2],
            "Gemini 2.5 Pro": [93.8, 85.4, 76.1, 71.0, 93.7, 84.1, 90.4, 80.6, 96.2, 81.8],
            "GPT-4o": [92.5, 83.1, 74.6, 68.9, 92.5, 82.4, 89.2, 78.7, 95.5, 80.4],
            "NExT-OMNI": [84.1, None, 65.2, None, 84.7, None, 81.5, None, 90.4, None],
            "Qwen3-Omni": [87.6, None, 70.3, None, 88.4, None, 84.9, None, 92.6, None],
            "Nemotron 3 Nano Omni (base)": [88.9, 71.4, 71.8, 60.3, 89.3, 73.5, 86.1, 71.2, 93.7, 72.8],
            "OmniVAL (Nemotron 3 Nano Omni)": [93.2, 84.6, 75.9, 70.4, 93.8, 83.9, 90.1, 80.2, 96.0, 81.6],
        },
    },
    "table_2_new_document_benchmarks": {
        "metrics": ["OCRBenchV2-En", "MMLongBench-Doc", "CharXiv"],
        "rows": {
            "Gemini 3.1 Pro": [82.7, 74.6, 79.8],
            "Gemini 2.5 Pro": [80.1, 72.3, 77.2],
            "GPT-4o": [76.5, 68.4, 73.5],
            "NExT-OMNI": [60.2, 51.4, 58.7],
            "Qwen3-Omni": [64.1, 55.3, 61.9],
            "Nemotron 3 Nano Omni (base)": [65.8, 57.5, 63.6],
            "OmniVAL (Nemotron 3 Nano Omni)": [74.1, 67.9, 72.4],
        },
    },
    "table_3_gui_video_audio_visual": {
        "metrics": ["ScreenSpot-Pro", "OSWorld", "Video-MME", "WorldSense", "DailyOmni"],
        "rows": {
            "Gemini 3.1 Pro": [72.8, 63.1, 84.7, 72.4, 86.9],
            "Gemini 2.5 Pro": [70.4, 60.5, 83.1, 70.6, 85.2],
            "GPT-4o": [65.7, 56.2, 79.4, 65.8, 80.7],
            "NExT-OMNI": [53.1, 42.6, 68.4, 50.8, 70.2],
            "Qwen3-Omni": [56.4, 45.9, 71.0, 53.7, 72.5],
            "Nemotron 3 Nano Omni (base)": [57.8, 47.4, 72.2, 55.4, 74.1],
            "OmniVAL (Nemotron 3 Nano Omni)": [64.3, 54.7, 77.8, 63.1, 80.4],
        },
    },
    "table_4_voice_asr_diarization": {
        "metrics": ["VoiceBench", "HF Open ASR WER", "DIHARD III DER"],
        "lower_is_better": [False, True, True],
        "rows": {
            "Gemini 3.1 Pro": [93.7, 4.12, 16.8],
            "Gemini 2.5 Pro": [92.6, 4.48, 17.6],
            "GPT-4o": [91.2, 5.04, 18.9],
            "NExT-OMNI": [84.5, 7.32, 24.7],
            "Qwen3-Omni": [87.8, 6.41, 22.9],
            "Nemotron 3 Nano Omni (base)": [89.4, 5.95, 21.4],
            "OmniVAL (Nemotron 3 Nano Omni)": [91.8, 4.86, 18.5],
        },
    },
    "table_6_validation_strategy": {
        "metrics": ["DocVQA_ANLS", "DocVQA_mAP", "OCRBenchV2", "ScreenSpot-Pro", "OSWorld", "Video-MME", "HF Open ASR WER", "DIHARD III DER"],
        "rows": {
            "No validation (120K unfiltered)": [89.7, 73.1, 67.4, 58.9, 48.9, 73.1, 5.62, 20.4],
            "MA-VAL Filter only": [91.8, 79.4, 71.0, 61.7, 51.8, 75.6, 5.21, 19.3],
            "MA-VAL Filter + Verifier": [93.2, 84.6, 74.1, 64.3, 54.7, 77.8, 4.86, 18.5],
        },
    },
    "table_7_module_4_ablation": {
        "metrics": ["DocVQA_ANLS", "DocVQA_mAP", "ScreenSpot-Pro", "Video-MME", "WorldSense", "DailyOmni"],
        "rows": {
            "MA-VAL without Module 4": [92.9, 84.1, 63.7, 76.5, 60.4, 77.0],
            "MA-VAL full": [93.2, 84.6, 64.3, 77.8, 63.1, 80.4],
        },
    },
    "table_8_training_strategy": {
        "metrics": ["DocVQA_ANLS", "DocVQA_mAP", "OCRBenchV2", "ScreenSpot-Pro", "Video-MME", "DailyOmni", "HF ASR WER", "DIHARD III DER"],
        "rows": {
            "Base Nemotron 3 Nano Omni": [88.9, 71.4, 65.8, 57.8, 72.2, 74.1, 5.95, 21.4],
            "Stage B1 only": [91.4, 78.9, 71.2, 61.4, 75.2, 77.6, 5.18, 19.4],
            "B1 + B2 (5 iterations)": [92.1, 81.0, 72.4, 62.7, 76.3, 78.7, 5.02, 19.0],
            "B1 + B2 (10 iterations)": [92.8, 83.4, 73.5, 63.6, 77.2, 79.6, 4.91, 18.7],
            "B1 + B2 (converged)": [93.2, 84.6, 74.1, 64.3, 77.8, 80.4, 4.86, 18.5],
        },
    },
    "table_9_data_scale": {
        "metrics": ["size", "DocVQA_ANLS", "DocVQA_mAP", "OCRBenchV2", "ScreenSpot-Pro", "Video-MME", "DailyOmni"],
        "rows": {
            "25%": ["22K", 89.7, 74.6, 67.5, 58.6, 73.4, 75.1],
            "50%": ["44K", 91.8, 80.3, 71.4, 61.5, 75.6, 77.5],
            "75%": ["66K", 92.6, 82.9, 73.0, 63.1, 76.9, 79.1],
            "100%": ["88K", 93.2, 84.6, 74.1, 64.3, 77.8, 80.4],
            "Unvalidated 120K": ["120K", 89.7, 73.1, 67.4, 58.9, 73.1, 75.0],
        },
    },
    "table_10_scaffolding_ablation": {
        "metrics": ["DocVQA_ANLS", "DocVQA_mAP", "ScreenSpot-Pro", "Video-MME", "HF Open ASR WER", "DIHARD III DER"],
        "rows": {
            "Full scaffolding": [93.2, 84.6, 64.3, 77.8, 4.86, 18.5],
            "without DB-ResNet": [91.4, 76.9, 64.3, 77.8, 4.86, 18.5],
            "without SAM 2.1": [92.5, 82.7, 60.6, 76.0, 4.86, 18.5],
            "without Whisper v3": [93.2, 84.6, 64.3, 75.1, 5.46, 19.7],
            "without pyannote": [93.2, 84.6, 64.3, 77.8, 4.86, 19.9],
            "without SigLIP": [93.2, 84.6, 64.3, 75.6, 4.86, 18.5],
            "No scaffolding": [89.6, 73.4, 58.4, 72.7, 5.74, 20.8],
        },
    },
    "table_11_teacher_ablation": {
        "metrics": ["DocVQA_mAP", "OCRBenchV2", "MMLongBench", "CharXiv", "ScreenSpot-Pro", "OSWorld", "Video-MME", "WorldSense", "DailyOmni"],
        "rows": {
            "Gemini 3.1 Pro": [84.6, 74.1, 67.9, 72.4, 64.3, 54.7, 77.8, 63.1, 80.4],
            "Claude 4.5 Sonnet": [83.5, 73.4, 67.0, 71.6, 63.6, 53.9, 77.0, 62.4, 79.7],
            "GPT-5": [83.1, 73.0, 66.5, 71.1, 63.2, 53.4, 76.6, 62.0, 79.3],
            "Gemini 2.5 Flash": [79.6, 70.3, 63.4, 68.1, 60.7, 50.9, 74.5, 59.8, 77.0],
            "GPT-4o": [78.9, 69.7, 62.9, 67.5, 60.1, 50.3, 74.0, 59.3, 76.5],
            "Qwen3-VL-235B-A22B-Thinking": [82.6, 72.5, 66.0, 70.6, 62.8, 53.0, 76.2, 61.5, 78.9],
            "Llama 4-400B-A17B": [81.9, 71.9, 65.4, 70.0, 62.3, 52.4, 75.7, 61.0, 78.3],
            "without teacher CoT": [71.4, 67.6, 60.1, 65.0, 59.4, 49.1, 73.4, 57.6, 75.0],
        },
    },
    "table_12_cross_modal_routing": {
        "metrics": ["Audio-only Q", "Visual-only Q", "AV-joint Q"],
        "rows": {
            "MA-VAL without Module 4": [76.3, 89.1, 73.6],
            "MA-VAL full": [91.7, 93.4, 88.2],
        },
    },
}


def paper_results() -> Dict[str, Any]:
    return deepcopy(PAPER_RESULTS)


def table_names() -> List[str]:
    return list(PAPER_RESULTS)
