import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReadmeTest(unittest.TestCase):
    def test_readme_routes_complete_run_order_and_states_execution_boundary(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for required in (
            "271 组 / 1075 query",
            "12787",
            "docs/commands/existing-models.md",
            "docs/commands/strong-baselines.md",
            "build_query_groups.py",
            "未运行耗时模型",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
