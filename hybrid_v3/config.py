import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur: Dict[str, Any] = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_override_value(raw: str) -> Any:
    s = raw.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "null"}:
        return None
    try:
        if s.startswith(("{", "[", "\"")):
            return json.loads(s)
    except Exception:
        pass
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def parse_overrides(overrides: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw_val = item.split("=", 1)
        _set_by_dotted_key(result, key.strip(), _parse_override_value(raw_val))
    return result


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_default_config_path() -> Path:
    return Path(__file__).with_name("configs").joinpath("default.json")


def load_config(
    section: str,
    *,
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    kv_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Precedence: default.json < --config < CLI kwargs < --override key=value
    Returns merged config for `common` + `section`.
    """
    cfg = load_json(get_default_config_path())

    if config_path is not None:
        cfg = _deep_update(cfg, load_json(config_path))

    common = deepcopy(cfg.get("common", {}))
    sect = deepcopy(cfg.get(section, {}))
    merged = _deep_update(common, sect)

    if cli_overrides:
        merged = _deep_update(merged, deepcopy(cli_overrides))

    if kv_overrides:
        merged = _deep_update(merged, parse_overrides(kv_overrides))

    return merged
