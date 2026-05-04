"""Command-line interface for local OmniVAL workflows."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .benchmarks import BenchmarkRegistry
from .config import default_config
from .models import MODEL_CATALOG, GroundTruthTemplateTeacher, MemorizingOmniStudent
from .pipeline import OmniVALPipeline
from .results import paper_results, table_names
from .sample_data import create_synthetic_pool


def _cmd_smoke(args: argparse.Namespace) -> int:
    examples = create_synthetic_pool()
    pipeline = OmniVALPipeline(teacher=GroundTruthTemplateTeacher(), student=MemorizingOmniStudent())
    result = pipeline.run(examples)
    summary: dict[str, Any] = {
        "examples": len(examples),
        "accepted_phase_a": len(result.phase_a.records),
        "rejected_phase_a": result.phase_a.rejected,
        "train_records": len(result.splits.train),
        "refinement_records": len(result.splits.refinement),
        "test_records": len(result.splits.test),
        "b2_iterations": result.stage_b2.iterations,
        "b2_score_history": [round(value, 4) for value in result.stage_b2.score_history],
        "b2_corrections": result.stage_b2.correction_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    registry = BenchmarkRegistry()
    payload = {
        "models": [spec.__dict__ for spec in MODEL_CATALOG],
        "benchmarks": [spec.__dict__ | {"modality": spec.modality.value, "grounding_kind": spec.grounding_kind.value} for spec in registry.all()],
        "config": default_config().to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_results(args: argparse.Namespace) -> int:
    results = paper_results()
    if args.table:
        results = {args.table: results[args.table]}
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniVAL OOP reference implementation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run the offline end-to-end synthetic pipeline")
    smoke.set_defaults(func=_cmd_smoke)

    catalog = subparsers.add_parser("catalog", help="Print all paper models, benchmarks, and default config")
    catalog.set_defaults(func=_cmd_catalog)

    results = subparsers.add_parser("paper-results", help="Print paper-reported main and ablation tables")
    results.add_argument("--table", choices=table_names(), help="Only print one result table")
    results.set_defaults(func=_cmd_results)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
