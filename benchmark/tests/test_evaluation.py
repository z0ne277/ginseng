import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ginseng_benchmark.cache import build_feature_cache, write_feature_cache_atomic
from ginseng_benchmark.evaluation import (
    evaluate_feature_cache,
    load_query_protocol,
    write_evaluation_json_atomic,
    write_per_query_csv_atomic,
)
from ginseng_benchmark.protocol import audit_sources
from ginseng_benchmark.query_groups import (
    build_query_protocol,
    write_query_protocol_atomic,
)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.library = self.root / "library"
        self.test = self.root / "test"
        self.gallery = self.root / "gallery"
        self.library.mkdir()
        (self.test / "1").mkdir(parents=True)
        self.gallery.mkdir()
        contents = {
            "negative.png": b"negative",
            "a.png": b"a",
            "b.png": b"b",
            "c.png": b"c",
        }
        (self.library / "negative.png").write_bytes(contents["negative.png"])
        for name in ("a.png", "b.png", "c.png"):
            (self.test / "1" / name).write_bytes(contents[name])
        for name, content in contents.items():
            (self.gallery / name).write_bytes(content)

        self.report = audit_sources(
            self.library, self.test, self.gallery, expected_groups=1
        )
        self.protocol_payload = build_query_protocol(
            self.test,
            self.gallery,
            dataset_manifest_sha256=self.report.manifest_sha256,
            gallery_count=4,
            expected_groups=1,
        )
        self.protocol_path = self.root / "query_groups.json"
        write_query_protocol_atomic(self.protocol_payload, self.protocol_path)

        feature_by_name = {
            "negative.png": np.array([0.0, 1.0], dtype=np.float32),
            "a.png": np.array([1.0, 0.0], dtype=np.float32),
            "b.png": np.array([0.995, 0.1], dtype=np.float32),
            "c.png": np.array([0.98, 0.2], dtype=np.float32),
        }
        for name in feature_by_name:
            feature_by_name[name] /= np.linalg.norm(feature_by_name[name])
        raw_paths = [self.gallery / name for name in ("c.png", "a.png", "negative.png", "b.png")]
        raw_features = np.stack([feature_by_name[path.name] for path in raw_paths])
        self.cache = build_feature_cache(
            raw_features,
            raw_paths,
            self.report,
            model_id="tiny",
            model_source="unit-test",
            feature_normalization="l2",
            preprocessing={"size": 224},
            tta={"enabled": False},
            environment={"python": "test"},
            expected_feature_dim=2,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_evaluation_excludes_self_and_preserves_fractional_recall(self):
        protocol = load_query_protocol(self.protocol_path)
        result = evaluate_feature_cache(
            self.cache,
            protocol,
            ks=(1, 2),
            block_size=2,
            bootstrap_iterations=20,
            bootstrap_seed=7,
        )

        self.assertEqual(result["metadata"]["ranking_scope"], "full")
        self.assertEqual(result["metadata"]["candidate_count_per_query"], 3)
        self.assertEqual(result["metadata"]["self_exclusion"], "cache_index_to_negative_infinity")
        self.assertEqual(result["aggregate"]["query_count"], 3)
        self.assertAlmostEqual(result["aggregate"]["macro"]["map"], 1.0)
        self.assertAlmostEqual(result["aggregate"]["macro"]["mrr"], 1.0)
        self.assertAlmostEqual(result["aggregate"]["macro"]["recall@1"], 0.5)
        self.assertAlmostEqual(result["aggregate"]["macro"]["recall@2"], 1.0)
        self.assertTrue(all(item["query_image"] not in item["top_results"] for item in result["per_query"]))
        json.dumps(result, allow_nan=False)

    def test_block_size_does_not_change_metrics_or_rankings(self):
        protocol = load_query_protocol(self.protocol_path)
        first = evaluate_feature_cache(
            self.cache, protocol, ks=(1, 2), block_size=1,
            bootstrap_iterations=10, bootstrap_seed=3,
        )
        second = evaluate_feature_cache(
            self.cache, protocol, ks=(1, 2), block_size=8,
            bootstrap_iterations=10, bootstrap_seed=3,
        )
        self.assertEqual(first, second)

    def test_ap_and_mrr_use_candidates_beyond_reported_top_k(self):
        feature_by_name = {
            "negative.png": np.array([0.98, 0.2], dtype=np.float32),
            "a.png": np.array([1.0, 0.0], dtype=np.float32),
            "b.png": np.array([0.995, 0.1], dtype=np.float32),
            "c.png": np.array([0.0, 1.0], dtype=np.float32),
        }
        for name in feature_by_name:
            feature_by_name[name] /= np.linalg.norm(feature_by_name[name])
        raw_paths = [self.gallery / name for name in feature_by_name]
        cache = build_feature_cache(
            np.stack([feature_by_name[path.name] for path in raw_paths]),
            raw_paths,
            self.report,
            model_id="full-ranking",
            model_source="unit-test",
            feature_normalization="l2",
            expected_feature_dim=2,
        )
        result = evaluate_feature_cache(
            cache,
            load_query_protocol(self.protocol_path),
            ks=(1,),
            bootstrap_iterations=10,
        )
        query_a = next(item for item in result["per_query"] if item["query_image"] == "a.png")
        self.assertEqual(query_a["top_results"][:3], ["b.png", "negative.png", "c.png"])
        self.assertAlmostEqual(query_a["map"], (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(query_a["mrr"], 1.0)
        self.assertAlmostEqual(query_a["recall@1"], 0.5)

    def test_ties_follow_canonical_manifest_order(self):
        tied = self.cache.features.copy()
        tied[:] = np.array([1.0, 0.0], dtype=np.float32)
        metadata = dict(self.cache.metadata)
        from ginseng_benchmark.cache import build_feature_cache
        raw_paths = [self.gallery / name for name in self.cache.paths]
        tied_cache = build_feature_cache(
            tied, raw_paths, self.report, model_id="ties", model_source="unit-test",
            feature_normalization="l2", expected_feature_dim=2,
        )
        result = evaluate_feature_cache(
            tied_cache, load_query_protocol(self.protocol_path), ks=(1,),
            block_size=3, bootstrap_iterations=10,
        )
        first = result["per_query"][0]
        expected = [name for name in tied_cache.paths if name != first["query_image"]][:3]
        self.assertEqual(first["top_results"], expected)
        self.assertEqual(result["metadata"]["tie_breaker"], "canonical_cache_order")

    def test_rejects_manifest_protocol_hash_missing_paths_and_zero_features(self):
        protocol = load_query_protocol(self.protocol_path)
        with self.subTest(kind="manifest"):
            bad = json.loads(json.dumps(protocol))
            bad["metadata"]["dataset_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest"):
                evaluate_feature_cache(self.cache, bad, bootstrap_iterations=10)

        with self.subTest(kind="protocol hash"):
            bad = json.loads(json.dumps(protocol))
            bad["query_groups"][0]["name"] = "changed"
            with self.assertRaisesRegex(ValueError, "protocol.*sha|name.*group_id"):
                evaluate_feature_cache(self.cache, bad, bootstrap_iterations=10)

        with self.subTest(kind="missing"):
            bad = json.loads(json.dumps(protocol))
            bad["query_groups"][0]["query_image"] = str(self.gallery / "missing.png")
            with self.assertRaisesRegex(ValueError, "protocol|cache|missing"):
                evaluate_feature_cache(self.cache, bad, bootstrap_iterations=10)

        with self.subTest(kind="zero"):
            zero_cache = type(self.cache)(
                features=np.zeros_like(self.cache.features),
                paths=self.cache.paths,
                metadata={**self.cache.metadata, "feature_normalization": "none"},
            )
            with self.assertRaisesRegex(ValueError, "zero"):
                evaluate_feature_cache(zero_cache, protocol, bootstrap_iterations=10)

    def test_protocol_loader_rejects_self_positive_duplicate_query_and_bad_metadata(self):
        payload = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        with self.subTest(kind="self"):
            bad = json.loads(json.dumps(payload))
            bad["query_groups"][0]["same_ginsengs"].append(bad["query_groups"][0]["query_image"])
            path = self.root / "bad-self.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "self"):
                load_query_protocol(path)
        with self.subTest(kind="duplicate"):
            bad = json.loads(json.dumps(payload))
            bad["query_groups"][1]["query_image"] = bad["query_groups"][0]["query_image"]
            path = self.root / "bad-duplicate.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_query_protocol(path)
        with self.subTest(kind="count"):
            bad = json.loads(json.dumps(payload))
            bad["metadata"]["query_count"] += 1
            path = self.root / "bad-count.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query_count"):
                load_query_protocol(path)

    def test_atomic_json_and_csv_writers(self):
        protocol = load_query_protocol(self.protocol_path)
        result = evaluate_feature_cache(
            self.cache, protocol, ks=(1, 2), bootstrap_iterations=10,
        )
        json_path = self.root / "result.json"
        csv_path = self.root / "per_query.csv"
        write_evaluation_json_atomic(result, json_path)
        write_per_query_csv_atomic(result["per_query"], csv_path)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["aggregate"]["query_count"], 3)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertIn("recall@1", rows[0])

        json_path.write_text("old", encoding="utf-8")
        with mock.patch("ginseng_benchmark.evaluation.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                write_evaluation_json_atomic(result, json_path)
        self.assertEqual(json_path.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(self.root.glob(".result.json.*.tmp")), [])


class EvaluateFeaturesScriptTest(unittest.TestCase):
    def test_cli_writes_json_and_csv_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "library"
            test = root / "test"
            gallery = root / "gallery"
            library.mkdir()
            (test / "1").mkdir(parents=True)
            gallery.mkdir()
            (library / "n.png").write_bytes(b"n")
            for name in ("a.png", "b.png"):
                (test / "1" / name).write_bytes(name.encode())
            for source in (library / "n.png", test / "1" / "a.png", test / "1" / "b.png"):
                (gallery / source.name).write_bytes(source.read_bytes())
            report = audit_sources(library, test, gallery, expected_groups=1)
            protocol_payload = build_query_protocol(
                test, gallery, report.manifest_sha256, 3, expected_groups=1
            )
            protocol_path = root / "query.json"
            write_query_protocol_atomic(protocol_payload, protocol_path)
            raw_paths = [gallery / "n.png", gallery / "a.png", gallery / "b.png"]
            raw_features = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32)
            cache = build_feature_cache(
                raw_features, raw_paths, report, model_id="cli", model_source="test",
                feature_normalization="l2", expected_feature_dim=2,
            )
            cache_path = root / "cache.npz"
            write_feature_cache_atomic(cache, cache_path)
            json_path = root / "result.json"
            csv_path = root / "result.csv"

            from scripts import evaluate_features
            exit_code = evaluate_features.main([
                "--cache", str(cache_path), "--query-groups", str(protocol_path),
                "--output", str(json_path), "--per-query-csv", str(csv_path),
                "--ks", "1,2", "--block-size", "1", "--bootstrap-iterations", "10",
            ])
            self.assertEqual(exit_code, 0)
            text = json_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), text)
            self.assertTrue(csv_path.is_file())

            with self.subTest(kind="colliding outputs"):
                exit_code = evaluate_features.main([
                    "--cache", str(cache_path), "--query-groups", str(protocol_path),
                    "--output", str(json_path), "--per-query-csv", str(json_path),
                    "--bootstrap-iterations", "10",
                ])
                self.assertNotEqual(exit_code, 0)

            with self.subTest(kind="input overwrite"):
                exit_code = evaluate_features.main([
                    "--cache", str(cache_path), "--query-groups", str(protocol_path),
                    "--output", str(protocol_path), "--bootstrap-iterations", "10",
                ])
                self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
