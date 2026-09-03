import unittest
from pathlib import Path

import torch

from scripts.extract_transreid import expected_transreid_dim, infer_checkpoint_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "dec55046fcdfadee14e2c28e2df89305d8f7557a"


class TransReIDAdapterTest(unittest.TestCase):
    def test_infers_classifier_and_sie_dimensions(self):
        state = {
            "module.classifier.weight": torch.zeros(271, 768),
            "module.base.sie_embed": torch.zeros(8, 1, 768),
        }
        layout = infer_checkpoint_layout(state)
        self.assertEqual(layout.num_classes, 271)
        self.assertEqual(layout.sie_embeddings, 8)
        self.assertEqual(layout.in_planes, 768)

    def test_expected_feature_dimension_accounts_for_jpm(self):
        self.assertEqual(expected_transreid_dim(in_planes=768, jpm=False), 768)
        self.assertEqual(expected_transreid_dim(in_planes=768, jpm=True), 3840)

    def test_setup_script_pins_official_repository(self):
        script = (REPO_ROOT / "scripts" / "setup_transreid.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("https://github.com/damo-cv/TransReID.git", script)
        self.assertIn(PINNED_COMMIT, script)
        self.assertIn("[switch]$DryRun", script)

    def test_runner_streams_all_conda_python_phases(self):
        script = (REPO_ROOT / "scripts" / "run_transreid.ps1").read_text(
            encoding="utf-8"
        )
        self.assertEqual(script.count('"run", "--no-capture-output"'), 3)
        self.assertEqual(script.count('"python", "-u"'), 3)


if __name__ == "__main__":
    unittest.main()
