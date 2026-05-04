import unittest

from omnival.data import BoundingBox, Grounding, GroundingKind, Modality, MultimodalExample, ScaffoldingEvidence, TextRegion, TimeSpan, Trace
from omnival.metrics import anls, cell_f1, iou_2d, temporal_iou
from omnival.validator import MAValidator, ValidationMode


class MetricTests(unittest.TestCase):
    def test_answer_and_grounding_metrics(self):
        self.assertEqual(anls("Total Due", "total due"), 1.0)
        self.assertAlmostEqual(iou_2d(BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)), 25 / 175)
        self.assertAlmostEqual(temporal_iou(TimeSpan(0, 1000), TimeSpan(500, 1500)), 1 / 3)
        from omnival.data import CellSpan

        self.assertEqual(cell_f1(CellSpan(0, 0, 0, 1), CellSpan(0, 1, 0, 2)), 0.5)


class ValidatorTests(unittest.TestCase):
    def test_accepts_exact_teacher_trace(self):
        box = BoundingBox(10, 20, 50, 70)
        example = MultimodalExample(
            example_id="doc-test",
            modality=Modality.DOCUMENT,
            input_uri="synthetic://doc",
            question="What is the total?",
            answer_gt="$45.99",
            grounding_gt=Grounding(GroundingKind.SPATIAL, Modality.DOCUMENT, box),
        )
        evidence = ScaffoldingEvidence(text_regions=[TextRegion("TOTAL $45.99", box, "DB-ResNet")])
        trace = Trace(
            reasoning="Modality-routing: the answer is in the document stream. I use the box region for TOTAL.",
            answer="$45.99",
            grounding=Grounding(GroundingKind.SPATIAL, Modality.DOCUMENT, box),
            modality_route=Modality.DOCUMENT,
        )
        feedback = MAValidator().evaluate(example, trace, evidence, ValidationMode.FILTER)
        self.assertTrue(feedback.score.accepted)
        self.assertGreater(feedback.score.overall, 0.99)

    def test_verifier_flags_wrong_modality_and_grounding(self):
        target_box = BoundingBox(10, 20, 50, 70)
        predicted_box = BoundingBox(100, 120, 150, 170)
        example = MultimodalExample(
            example_id="gui-test",
            modality=Modality.GUI,
            input_uri="synthetic://gui",
            question="Click save.",
            answer_gt="click",
            grounding_gt=Grounding(GroundingKind.SPATIAL, Modality.GUI, target_box),
        )
        trace = Trace(
            reasoning="Modality-routing: the answer is in the document stream.",
            answer="click",
            grounding=Grounding(GroundingKind.SPATIAL, Modality.DOCUMENT, predicted_box),
            modality_route=Modality.DOCUMENT,
        )
        feedback = MAValidator().evaluate(example, trace, ScaffoldingEvidence(), ValidationMode.VERIFIER)
        self.assertFalse(feedback.score.accepted)
        self.assertIn("Modality routing error", feedback.as_instruction())
        self.assertIsNotNone(feedback.corrected_trace)


if __name__ == "__main__":
    unittest.main()
