import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from ginseng_benchmark.cache import (
    build_feature_cache,
    load_feature_cache,
    load_trusted_torch_cache,
    write_feature_cache_atomic,
)
from ginseng_benchmark.protocol import audit_sources


class FeatureCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "library"
        self.test = self.root / "test"
        self.merged = self.root / "merged"
        self.library.mkdir()
        (self.test / "group-1").mkdir(parents=True)
        self.merged.mkdir()
        (self.library / "a.png").write_bytes(b"a")
        (self.test / "group-1" / "b.png").write_bytes(b"b")
        (self.merged / "a.png").write_bytes(b"a")
        (self.merged / "b.png").write_bytes(b"b")
        self.report = audit_sources(
            self.library,
            self.test,
            self.merged,
            expected_groups=1,
        )
        self.output = self.root / "cache.npz"

    def _build(self, **overrides):
        arguments = {
            "raw_features": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "raw_paths": [self.merged / "b.png", self.merged / "a.png"],
            "report": self.report,
            "model_id": "tiny-model",
            "model_source": "unit-test",
            "feature_normalization": "l2",
            "preprocessing": {"resize": 224},
            "tta": {"enabled": False},
            "environment": {"torch": "1.13.1"},
        }
        arguments.update(overrides)
        return build_feature_cache(**arguments)

    def test_reorders_raw_rows_to_canonical_manifest_basename_order(self):
        cache = self._build()

        self.assertEqual(cache.paths, ("a.png", "b.png"))
        np.testing.assert_array_equal(
            cache.features,
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(cache.features.dtype, np.float32)
        self.assertTrue(cache.features.flags.c_contiguous)
        self.assertEqual(cache.metadata["dataset_manifest_sha256"], self.report.manifest_sha256)
        self.assertEqual(cache.metadata["num_images"], 2)
        self.assertEqual(cache.metadata["feature_dim"], 2)

    def test_accepts_basenames_by_mapping_them_to_the_merged_root(self):
        cache = self._build(raw_paths=["b.png", "a.png"])

        self.assertEqual(cache.paths, ("a.png", "b.png"))

    def test_rejects_raw_path_outside_merged_root(self):
        escaped = self.root / "outside.png"
        escaped.write_bytes(b"outside")

        with self.assertRaisesRegex(ValueError, "outside.*merged|escape"):
            self._build(raw_paths=[self.merged / "b.png", escaped])

    def test_rejects_casefold_duplicate_raw_paths(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._build(raw_paths=["a.png", "A.PNG"])

    def test_rejects_missing_manifest_path(self):
        with self.assertRaisesRegex(ValueError, "missing|set mismatch"):
            self._build(
                raw_features=np.array([[1.0, 0.0]], dtype=np.float32),
                raw_paths=["a.png"],
            )

    def test_rejects_extra_raw_path(self):
        (self.merged / "extra.png").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra|set mismatch"):
            self._build(
                raw_features=np.eye(3, dtype=np.float32),
                raw_paths=["a.png", "b.png", "extra.png"],
            )

    def test_rejects_feature_path_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "count"):
            self._build(raw_features=np.array([[1.0, 0.0]], dtype=np.float32))

    def test_rejects_non_matrix_zero_width_and_wrong_expected_dimension(self):
        for features, pattern in (
            (np.ones(2, dtype=np.float32), "two-dimensional|2D"),
            (np.empty((2, 0), dtype=np.float32), "dimension|width"),
        ):
            with self.subTest(shape=features.shape):
                with self.assertRaisesRegex(ValueError, pattern):
                    self._build(raw_features=features)

        with self.assertRaisesRegex(ValueError, "feature dimension"):
            self._build(expected_feature_dim=3)

    def test_rejects_non_numeric_bool_and_complex_features(self):
        invalid_features = (
            np.array([["x", "y"], ["z", "w"]]),
            np.array([[True, False], [False, True]]),
            np.array([[1 + 1j, 0], [0, 1]], dtype=np.complex64),
            np.array([[object(), object()], [object(), object()]], dtype=object),
        )
        for features in invalid_features:
            with self.subTest(dtype=str(features.dtype)):
                with self.assertRaisesRegex(ValueError, "dtype|numeric|real"):
                    self._build(raw_features=features)

    def test_rejects_nan_and_infinity(self):
        for invalid in (math.nan, math.inf, -math.inf):
            features = np.array([[invalid, 0.0], [0.0, 1.0]], dtype=np.float32)
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self._build(raw_features=features)

    def test_l2_mode_rejects_zero_or_not_normalized_rows(self):
        for features in (
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ):
            with self.subTest(features=features.tolist()):
                with self.assertRaisesRegex(ValueError, "L2|l2|norm"):
                    self._build(raw_features=features)

    def test_none_mode_accepts_finite_non_normalized_features(self):
        cache = self._build(
            raw_features=np.array([[0.0, 3.0], [2.0, 0.0]], dtype=np.float64),
            feature_normalization="none",
        )

        self.assertEqual(cache.features.dtype, np.float32)
        np.testing.assert_array_equal(cache.features, [[2.0, 0.0], [0.0, 3.0]])

    def test_rejects_sensitive_non_json_nonfinite_and_absolute_metadata(self):
        cases = (
            ({"api_token": "do-not-store"}, "sensitive"),
            ({"nested": {"PASSWORD": "do-not-store"}}, "sensitive"),
            ({"bad": {1, 2}}, "JSON"),
            ({"bad": math.nan}, "finite|JSON"),
            ({"weights": str((self.root / "weights.bin").resolve())}, "absolute"),
        )
        for preprocessing, pattern in cases:
            with self.subTest(preprocessing=repr(preprocessing)):
                with self.assertRaisesRegex(ValueError, pattern):
                    self._build(preprocessing=preprocessing)

    def test_checkpoint_metadata_contains_only_basename_and_digest(self):
        checkpoint = self.root / "private" / "weights.bin"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")

        cache = self._build(checkpoint=checkpoint)

        self.assertEqual(cache.metadata["checkpoint"]["name"], "weights.bin")
        self.assertEqual(len(cache.metadata["checkpoint"]["sha256"]), 64)
        self.assertNotIn(str(checkpoint.parent), json.dumps(cache.metadata))

    def test_round_trip_uses_strict_npz_schema_and_hashes(self):
        cache = self._build()
        write_feature_cache_atomic(cache, self.output)

        with np.load(self.output, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {"features", "paths", "metadata_json"})
            self.assertEqual(archive["features"].dtype, np.float32)
            self.assertEqual(archive["paths"].dtype.kind, "U")
            self.assertEqual(archive["metadata_json"].dtype.kind, "U")
            self.assertEqual(archive["metadata_json"].ndim, 0)

        loaded = load_feature_cache(
            self.output,
            expected_dataset_manifest_sha256=self.report.manifest_sha256,
            expected_paths=("a.png", "b.png"),
        )
        np.testing.assert_array_equal(loaded.features, cache.features)
        self.assertEqual(loaded.paths, cache.paths)
        self.assertEqual(loaded.metadata, cache.metadata)

    def test_load_rejects_legacy_cache_without_manifest(self):
        metadata = {
            "schema_version": 1,
            "model_id": "legacy",
            "num_images": 2,
            "feature_dim": 2,
        }
        np.savez(
            self.output,
            features=np.eye(2, dtype=np.float32),
            paths=np.array(["a.png", "b.png"]),
            metadata_json=np.array(json.dumps(metadata)),
        )

        with self.assertRaisesRegex(ValueError, "dataset_manifest_sha256|metadata"):
            load_feature_cache(self.output)

    def test_load_rejects_object_arrays_without_enabling_pickle(self):
        np.savez(
            self.output,
            features=np.eye(2, dtype=np.float32),
            paths=np.array(["a.png", "b.png"], dtype=object),
            metadata_json=np.array("{}"),
        )

        with self.assertRaisesRegex(ValueError, "object|pickle|Unicode"):
            load_feature_cache(self.output)

    def test_load_rejects_non_c_contiguous_feature_archive(self):
        cache = self._build()
        fortran_features = np.asfortranarray(cache.features)
        self.assertFalse(fortran_features.flags.c_contiguous)
        np.savez(
            self.output,
            features=fortran_features,
            paths=np.array(cache.paths),
            metadata_json=np.array(json.dumps(cache.metadata)),
        )

        with self.assertRaisesRegex(ValueError, "C contiguous"):
            load_feature_cache(self.output)

    def test_load_rejects_feature_and_path_hash_tampering(self):
        cache = self._build()
        write_feature_cache_atomic(cache, self.output)
        metadata_json = json.dumps(cache.metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

        with self.subTest(kind="features"):
            np.savez(
                self.output,
                features=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                paths=np.array(cache.paths),
                metadata_json=np.array(metadata_json),
            )
            with self.assertRaisesRegex(ValueError, "features_sha256"):
                load_feature_cache(self.output)

        with self.subTest(kind="paths"):
            np.savez(
                self.output,
                features=cache.features,
                paths=np.array(["b.png", "a.png"]),
                metadata_json=np.array(metadata_json),
            )
            with self.assertRaisesRegex(ValueError, "paths_sha256"):
                load_feature_cache(self.output)

    def test_load_rejects_expected_manifest_and_path_mismatch(self):
        write_feature_cache_atomic(self._build(), self.output)

        with self.assertRaisesRegex(ValueError, "manifest"):
            load_feature_cache(
                self.output,
                expected_dataset_manifest_sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "expected paths|path"):
            load_feature_cache(
                self.output,
                expected_paths=("b.png", "a.png"),
            )

    def test_load_rejects_casefold_duplicate_paths_even_with_matching_hash(self):
        cache = self._build()
        metadata = dict(cache.metadata)
        paths = ("a.png", "A.PNG")
        canonical = json.dumps(paths, ensure_ascii=False, separators=(",", ":"))
        import hashlib
        metadata["paths_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        np.savez(
            self.output,
            features=cache.features,
            paths=np.array(paths),
            metadata_json=np.array(json.dumps(metadata)),
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_feature_cache(self.output)

    def test_atomic_replace_failure_preserves_old_output_and_cleans_temp(self):
        self.output.write_bytes(b"old-cache")

        with mock.patch(
            "ginseng_benchmark.cache.os.replace",
            side_effect=OSError("simulated replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "replace failure"):
                write_feature_cache_atomic(self._build(), self.output)

        self.assertEqual(self.output.read_bytes(), b"old-cache")
        self.assertEqual(list(self.root.glob(".cache.npz.*.tmp")), [])

    def test_serialization_failure_preserves_old_output_and_cleans_temp(self):
        self.output.write_bytes(b"old-cache")

        with mock.patch(
            "ginseng_benchmark.cache.np.savez_compressed",
            side_effect=OSError("simulated serialization failure"),
        ):
            with self.assertRaisesRegex(OSError, "serialization failure"):
                write_feature_cache_atomic(self._build(), self.output)

        self.assertEqual(self.output.read_bytes(), b"old-cache")
        self.assertEqual(list(self.root.glob(".cache.npz.*.tmp")), [])

    def test_requires_npz_output_suffix(self):
        with self.assertRaisesRegex(ValueError, "npz"):
            write_feature_cache_atomic(self._build(), self.root / "cache.pt")


class TrustedTorchCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("torch is unavailable") from error

    def test_requires_explicit_trust_and_returns_cpu_numpy_plus_paths(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_cache = Path(temporary_directory) / "raw.pt"
            torch.save(
                {"features": torch.eye(2), "paths": ["a.png", "b.png"], "ignored": {"token": "no"}},
                raw_cache,
            )
            with self.assertRaisesRegex(ValueError, "pickle|trusted"):
                load_trusted_torch_cache(raw_cache, trusted_local_pt=False)

            features, paths = load_trusted_torch_cache(
                raw_cache,
                trusted_local_pt=True,
            )

        self.assertIsInstance(features, np.ndarray)
        np.testing.assert_array_equal(features, np.eye(2, dtype=np.float32))
        self.assertEqual(paths, ("a.png", "b.png"))

    def test_rejects_wrong_suffix_or_missing_required_keys(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wrong_suffix = root / "raw.pth"
            torch.save({"features": torch.eye(1), "paths": ["a.png"]}, wrong_suffix)
            with self.assertRaisesRegex(ValueError, "\.pt"):
                load_trusted_torch_cache(wrong_suffix, trusted_local_pt=True)

            raw_cache = root / "raw.pt"
            torch.save({"features": torch.eye(1)}, raw_cache)
            with self.assertRaisesRegex(ValueError, "features.*paths|paths"):
                load_trusted_torch_cache(raw_cache, trusted_local_pt=True)


class StampFeatureCacheScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("torch is unavailable") from error

    def setUp(self):
        import torch

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.library = self.root / "library"
        self.test = self.root / "test"
        self.merged = self.root / "merged"
        self.library.mkdir()
        (self.test / "group-1").mkdir(parents=True)
        self.merged.mkdir()
        (self.library / "a.png").write_bytes(b"a")
        (self.test / "group-1" / "b.png").write_bytes(b"b")
        (self.merged / "a.png").write_bytes(b"a")
        (self.merged / "b.png").write_bytes(b"b")
        self.env_path = self.root / ".env"
        self.env_path.write_text(
            f"LIBRARY_BINARY={self.library}\n"
            f"TEST_BINARY_ROOT={self.test}\n"
            f"MERGED_GALLERY={self.merged}\n"
            "HF_TOKEN=not-for-output\n",
            encoding="utf-8",
        )
        self.raw_cache = self.root / "raw.pt"
        torch.save(
            {
                "features": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                "paths": [str(self.merged / "b.png"), str(self.merged / "a.png")],
                "metadata": {"api_key": "must-be-ignored"},
            },
            self.raw_cache,
        )
        self.output = self.root / "stamped.npz"
        self.script = Path(__file__).parents[1] / "scripts" / "stamp_feature_cache.py"

    def _run(self, *extra_arguments):
        return subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--env",
                str(self.env_path),
                "--raw-cache",
                str(self.raw_cache),
                "--output",
                str(self.output),
                "--model-id",
                "tiny-model",
                "--model-source",
                "unit-test",
                "--feature-normalization",
                "l2",
                "--expected-groups",
                "1",
                "--expected-library-count",
                "1",
                "--expected-test-count",
                "1",
                "--expected-merged-count",
                "2",
                *extra_arguments,
            ],
            cwd=self.script.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_refuses_pt_without_explicit_trust_and_does_not_write(self):
        result = self._run()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pickle", result.stderr.casefold())
        self.assertFalse(self.output.exists())
        self.assertNotIn("not-for-output", result.stdout + result.stderr)

    def test_audit_mismatch_or_count_failure_does_not_write(self):
        (self.merged / "a.png").write_bytes(b"mismatch")
        result = self._run("--trusted-local-pt")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mismatch", result.stderr.casefold())
        self.assertFalse(self.output.exists())

        (self.merged / "a.png").write_bytes(b"a")
        result = self._run("--trusted-local-pt", "--expected-library-count", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("library count", result.stderr.casefold())
        self.assertFalse(self.output.exists())

    def test_stamps_tiny_fixture_and_prints_safe_summary(self):
        result = self._run(
            "--trusted-local-pt",
            "--expected-feature-dim",
            "2",
            "--preprocessing-json",
            '{"resize":224}',
            "--tta-json",
            '{"enabled":false}',
            "--environment-json",
            '{"torch":"1.13.1"}',
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.is_file())
        for field in (
            "model=tiny-model",
            "count=2",
            "dim=2",
            "manifest_sha256=",
            "features_sha256=",
            "paths_sha256=",
        ):
            self.assertIn(field, result.stdout)
        self.assertNotIn(str(self.raw_cache), result.stdout + result.stderr)
        self.assertNotIn("not-for-output", result.stdout + result.stderr)
        loaded = load_feature_cache(self.output, expected_paths=("a.png", "b.png"))
        self.assertEqual(loaded.metadata["preprocessing"], {"resize": 224})

    def test_rejects_non_object_json_without_writing(self):
        result = self._run(
            "--trusted-local-pt",
            "--preprocessing-json",
            "[]",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("object", result.stderr.casefold())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
