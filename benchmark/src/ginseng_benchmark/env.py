"""Small, dependency-free helpers for reading local environment files."""

from pathlib import Path
from typing import Dict


def _strip_paired_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path) -> Dict[str, str]:
    """Load key-value pairs from *path* without changing ``os.environ``."""
    values: Dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig") as env_file:
        for line_number, raw_line in enumerate(env_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(
                    f"Invalid environment entry on line {line_number}: missing '='"
                )

            raw_key, raw_value = line.split("=", 1)
            key = _strip_paired_quotes(raw_key.strip())
            value = _strip_paired_quotes(raw_value.strip())
            values[key] = value

    return values
