"""No model downloads or GPU required."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import parse_visual_response, run


class PipelineTests(unittest.TestCase):
    def test_valid_model_json(self):
        result = parse_visual_response('''```json
        {"page_kind":"form","has_handwriting":false,
        "brief_visual_description":"A printed form", "needs_human_review":false}
        ```''')
        self.assertEqual(result["page_kind"], "form")
        self.assertFalse(result["needs_human_review"])
        self.assertEqual(result["inference_status"], "ok")

    def test_invalid_model_json(self):
        result = parse_visual_response("not JSON")
        self.assertTrue(result["needs_human_review"])
        self.assertEqual(result["inference_status"], "invalid_model_output")

    def test_cpu_parsing_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "in"
            dest = Path(temp) / "out"
            source.mkdir()
            (source / "example.txt").write_text("Synthetic text", encoding="utf-8")
            (source / "table.csv").write_text("id,x\na,1\nb,2\n", encoding="utf-8")
            result = run(source, dest, None)
            self.assertEqual(len(result), 2)
            self.assertTrue((dest / "page_features.csv").exists())
            self.assertEqual(int(result.loc[result.file_type == ".csv", "tabular_rows"].iloc[0]), 2)


if __name__ == "__main__":
    unittest.main()
