import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ginseng_benchmark.ssl_models import (
    DinoHead,
    DinoModel,
    SimSiamModel,
    VICRegModel,
    ema_update,
    update_dino_center,
)
from ginseng_benchmark.ssl_runtime import seed_torch_checkpoint_cache
from ginseng_benchmark.ssl_training import (
    dino_cross_view_loss,
    load_ssl_config,
    negative_cosine_similarity,
    validate_image_only_csv,
    vicreg_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SelfSupervisedConfigTest(unittest.TestCase):
    def test_project_config_contains_only_unlabeled_training_methods(self):
        config = load_ssl_config(REPO_ROOT / "configs" / "self_supervised_models.json")

        self.assertEqual(
            {model.model_id for model in config.models},
            {"simsiam_r50", "vicreg_r50", "dino_vits16"},
        )
        self.assertTrue(all(not model.requires_identity_labels for model in config.models))
        self.assertEqual(
            {model.algorithm for model in config.models},
            {"simsiam", "vicreg", "dino"},
        )

    def test_rejects_a_supervised_method_in_unlabeled_config(self):
        payload = {
            "schema_version": 1,
            "protocol_tag": "271_1075_unlabeled",
            "models": [
                {
                    "id": "triplet",
                    "algorithm": "triplet",
                    "backbone": "resnet50",
                    "conda_env": "gsam",
                    "feature_dim": 2048,
                    "requires_identity_labels": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity labels"):
                load_ssl_config(path)

    def test_powershell_dry_run_builds_train_extract_stamp_and_evaluate_commands(self):
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "scripts" / "run_self_supervised_baselines.ps1"),
            "-Models",
            "simsiam_r50",
            "-Phase",
            "all",
            "-DryRun",
        ]

        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("train_ssl_baseline.py", completed.stdout)
        self.assertIn("extract_ssl_checkpoint.py", completed.stdout)
        self.assertIn("stamp_feature_cache.py", completed.stdout)
        self.assertIn("evaluate_features.py", completed.stdout)
        self.assertIn("-n gsam", completed.stdout)
        self.assertNotIn("vicreg_r50", completed.stdout)

    def test_existing_torch_weight_is_copied_into_project_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy"
            target = root / "project"
            source.mkdir()
            content = b"pretrained-weight"
            digest = hashlib.sha256(content).hexdigest()
            filename = f"resnet50-{digest[:8]}.pth"
            (source / filename).write_bytes(content)

            copied = seed_torch_checkpoint_cache(
                target,
                source,
                filenames=(filename,),
            )

            self.assertEqual(copied, (target / filename,))
            self.assertEqual((target / filename).read_bytes(), content)


class ImageOnlyCsvTest(unittest.TestCase):
    def test_accepts_unique_existing_image_paths_and_returns_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / "a.jpg", root / "b.jpg"]
            for image in images:
                image.write_bytes(b"image")
            csv_path = root / "train.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image"])
                writer.writeheader()
                for image in images:
                    writer.writerow({"image": str(image)})

            paths = validate_image_only_csv(csv_path)

            self.assertEqual(paths, tuple(images))

    def test_rejects_identity_or_label_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_text("image,identity\na.jpg,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only the image column"):
                validate_image_only_csv(path, require_files=False)


class SelfSupervisedLossTest(unittest.TestCase):
    def test_simsiam_negative_cosine_prefers_aligned_pairs(self):
        aligned = torch.eye(4)
        opposite = -aligned

        aligned_loss = negative_cosine_similarity(aligned, aligned)
        opposite_loss = negative_cosine_similarity(opposite, aligned)

        self.assertLess(aligned_loss.item(), opposite_loss.item())
        self.assertAlmostEqual(aligned_loss.item(), -1.0, places=6)

    def test_vicreg_is_finite_and_backpropagates(self):
        first = torch.randn(8, 6, requires_grad=True)
        second = torch.randn(8, 6, requires_grad=True)

        loss, components = vicreg_loss(first, second)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(components), {"invariance", "variance", "covariance"})
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)

    def test_dino_cross_view_excludes_matching_views(self):
        student = [torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 10.0]])]
        teacher = [torch.tensor([[10.0, 0.0]]), torch.tensor([[0.0, 10.0]])]
        center = torch.zeros(1, 2)

        loss = dino_cross_view_loss(
            student,
            teacher,
            center=center,
            student_temperature=0.1,
            teacher_temperature=0.04,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 1.0)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(12, 8)

    def forward(self, images):
        return self.linear(self.flatten(images))


class SelfSupervisedModelTest(unittest.TestCase):
    def test_simsiam_returns_predictions_and_detached_targets(self):
        model = SimSiamModel(
            TinyEncoder(),
            feature_dim=8,
            projection_dim=6,
            projection_hidden_dim=10,
            prediction_hidden_dim=4,
        )
        first = torch.randn(3, 3, 2, 2)
        second = torch.randn(3, 3, 2, 2)

        prediction_first, prediction_second, target_first, target_second = model(
            first, second
        )

        self.assertEqual(prediction_first.shape, (3, 6))
        self.assertEqual(prediction_second.shape, (3, 6))
        self.assertFalse(target_first.requires_grad)
        self.assertFalse(target_second.requires_grad)
        self.assertEqual(model.encode(first).shape, (3, 8))

    def test_vicreg_returns_two_projected_views(self):
        model = VICRegModel(
            TinyEncoder(),
            feature_dim=8,
            projection_dim=6,
            projection_hidden_dim=10,
        )
        first, second = model(
            torch.randn(3, 3, 2, 2),
            torch.randn(3, 3, 2, 2),
        )

        self.assertEqual(first.shape, (3, 6))
        self.assertEqual(second.shape, (3, 6))

    def test_ema_update_moves_target_towards_online_parameters(self):
        online = nn.Linear(2, 2, bias=False)
        target = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            online.weight.fill_(2.0)
            target.weight.zero_()

        ema_update(target, online, momentum=0.75)

        self.assertTrue(torch.allclose(target.weight, torch.full_like(target.weight, 0.5)))

    def test_dino_head_and_center_update_have_stable_shapes(self):
        head = DinoHead(
            input_dim=8,
            output_dim=5,
            hidden_dim=12,
            bottleneck_dim=4,
        )
        logits = head(torch.randn(3, 8))
        old_center = torch.zeros(1, 5)

        new_center = update_dino_center(old_center, logits, momentum=0.9)

        self.assertEqual(logits.shape, (3, 5))
        self.assertEqual(new_center.shape, (1, 5))
        self.assertFalse(new_center.requires_grad)

    def test_dino_model_uses_all_student_views_and_two_teacher_views(self):
        model = DinoModel(
            TinyEncoder(),
            feature_dim=8,
            output_dim=5,
            hidden_dim=12,
            bottleneck_dim=4,
        )
        views = [torch.randn(3, 3, 2, 2) for _ in range(4)]

        student_logits, teacher_logits = model(views)

        self.assertEqual(len(student_logits), 4)
        self.assertEqual(len(teacher_logits), 2)
        self.assertTrue(all(item.shape == (3, 5) for item in student_logits))
        self.assertTrue(all(item.shape == (3, 5) for item in teacher_logits))
        self.assertEqual(model.encode(views[0]).shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
