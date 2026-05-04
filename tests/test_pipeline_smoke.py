import unittest

from omnival.benchmarks import BenchmarkRegistry
from omnival.models import GroundTruthTemplateTeacher, MemorizingOmniStudent
from omnival.pipeline import OmniVALPipeline
from omnival.sample_data import create_synthetic_pool


class PipelineSmokeTests(unittest.TestCase):
    def test_catalog_contains_all_paper_benchmarks(self):
        registry = BenchmarkRegistry()
        self.assertEqual(len(registry.names()), 16)
        self.assertIn("DIHARD III", registry.names())
        self.assertIn("ScreenSpot-Pro", registry.names())

    def test_offline_pipeline_runs(self):
        examples = create_synthetic_pool()
        result = OmniVALPipeline(
            teacher=GroundTruthTemplateTeacher(),
            student=MemorizingOmniStudent(),
        ).run(examples)
        self.assertEqual(len(result.phase_a.records), len(examples))
        self.assertEqual(result.phase_a.rejected, 0)
        self.assertGreater(len(result.splits.train), 0)
        self.assertGreater(result.stage_b2.iterations, 0)
        self.assertGreaterEqual(result.stage_b2.score_history[-1], 0.85)


if __name__ == "__main__":
    unittest.main()
