from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SCRIPT_PATHS = (
    Path(__file__).parents[1] / "gradcam_visualization.py",
    Path(__file__).parents[2] / "single_topo_scale_local" / "gradcam_visualization.py",
)


def load_visualizer(script_path: Path, module_name: str):
    sys.path.insert(0, str(script_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def extract_word_boxes(pdf_path: Path):
    completed = subprocess.run(
        ["pdftotext", "-bbox", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    root = ET.fromstring(completed.stdout)
    boxes = {}
    for element in root.iter():
        if not element.tag.endswith("word") or element.text is None:
            continue
        boxes[element.text] = (
            float(element.attrib["xMin"]),
            float(element.attrib["xMax"]),
            float(element.attrib["yMin"]),
            float(element.attrib["yMax"]),
        )
    return boxes


class SummaryRowLabelTest(unittest.TestCase):
    def test_panel_markers_are_ascii_and_center_aligned(self):
        """Changing a marker or its shared horizontal alignment must break this test."""
        image = np.zeros((20, 30, 3), dtype=np.float32)
        results = [
            {
                "group": "test",
                "query_rgb": image,
                "reference_rgb": image,
                "overlay": image,
            }
        ]

        for index, script_path in enumerate(SCRIPT_PATHS):
            with self.subTest(script=str(script_path)):
                module = load_visualizer(script_path, f"gradcam_visualization_{index}")
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_prefix = Path(temp_dir) / "summary"
                    module.build_summary_figure(results, output_prefix)
                    boxes = extract_word_boxes(output_prefix.with_suffix(".pdf"))

                markers = ["(a)", "(b)", "(c)"]
                self.assertTrue(all(marker in boxes for marker in markers), boxes.keys())
                x_centers = [
                    (boxes[marker][0] + boxes[marker][1]) / 2.0 for marker in markers
                ]
                self.assertLessEqual(max(x_centers) - min(x_centers), 0.25)


if __name__ == "__main__":
    unittest.main()
