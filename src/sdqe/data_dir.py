"""``--data_dir`` 폴더에서 ``<prefix>_origin`` / ``<prefix>_synth`` 데이터 쌍을 찾습니다.

규칙
- 폴더 바로 아래(하위 폴더 제외)에서 ``<prefix>_origin.<확장자>`` 와 ``<prefix>_synth.<확장자>`` 를
  찾습니다. 확장자는 ``config.DATA_DIR_EXTENSIONS`` (csv, parquet, xlsx)이고,
  대소문자는 구분하지 않습니다.
- 쌍이 하나면 자동 선택하고, 여럿이면 ``--prefix`` 로 고르도록 에러를 냅니다.
- 같은 prefix·역할에 확장자만 다른 파일이 둘 이상이면 어느 쪽인지 모호하므로 에러를 냅니다.
- 결과 폴더는 ``<data_dir>/<prefix>_results`` 입니다.
- Excel 잠금 파일(``~$`` 로 시작)은 무시합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdqe import config


class DataDirError(ValueError):
    """폴더에서 데이터 쌍을 확정할 수 없을 때 발생합니다."""


@dataclass(frozen=True)
class DataPair:
    """폴더에서 찾은 데이터 쌍."""

    prefix: str
    original: Path
    synthetic: Path | None
    output_dir: Path


def _scan(data_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """``{prefix: {"origin": [...], "synth": [...]}}`` 형태로 후보 파일을 모읍니다."""
    found: dict[str, dict[str, list[Path]]] = {}
    roles = {suffix.lower(): role for role, suffix in config.DATA_DIR_SUFFIXES.items()}
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in config.DATA_DIR_EXTENSIONS:
            continue
        stem = path.stem
        for suffix, role in roles.items():
            if stem.lower().endswith(suffix):
                prefix = stem[: len(stem) - len(suffix)]
                found.setdefault(prefix, {"origin": [], "synth": []})[role].append(path)
                break
    return found


def _single(paths: list[Path], prefix: str, role: str) -> Path:
    if len(paths) > 1:
        names = ", ".join(p.name for p in paths)
        raise DataDirError(
            f"Ambiguous {role} files for prefix '{prefix}': {names}. Keep only one format."
        )
    return paths[0]


def results_dir_name(prefix: str) -> str:
    """prefix 에 대응하는 결과 폴더 이름."""
    return f"{prefix}{config.DATA_DIR_RESULTS_SUFFIX}" if prefix else "results"


def find_data_pair(
    data_dir: str | Path, prefix: str | None = None, need_synthetic: bool = True
) -> DataPair:
    """폴더에서 데이터 쌍을 찾습니다.

    Parameters
    ----------
    data_dir
        ``<prefix>_origin`` / ``<prefix>_synth`` 파일이 있는 폴더.
    prefix
        폴더에 쌍이 여럿일 때 고를 prefix. ``None`` 이면 유일한 쌍을 자동 선택합니다.
    need_synthetic
        ``False`` 이면(``threshold``) 원본만 있어도 됩니다.

    Raises
    ------
    DataDirError
        폴더가 없거나, 쌍을 찾지 못했거나, 쌍이 여럿인데 prefix 가 없을 때.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise DataDirError(f"--data_dir {root} is not a directory.")
    found = _scan(root)
    origin_sfx = config.DATA_DIR_SUFFIXES["origin"]
    synth_sfx = config.DATA_DIR_SUFFIXES["synth"]

    def usable(roles: dict[str, list[Path]]) -> bool:
        return bool(roles["origin"]) and (bool(roles["synth"]) or not need_synthetic)

    if prefix is not None:
        roles = found.get(prefix)
        if roles is None or not roles["origin"]:
            raise DataDirError(f"No '{prefix}{origin_sfx}.*' file in {root}.")
        if need_synthetic and not roles["synth"]:
            raise DataDirError(f"No '{prefix}{synth_sfx}.*' file in {root}.")
        chosen = prefix
    else:
        candidates = sorted(p for p, roles in found.items() if usable(roles))
        if not candidates:
            raise DataDirError(_not_found_message(root, found, need_synthetic))
        if len(candidates) > 1:
            listed = ", ".join(f"'{c}'" for c in candidates)
            raise DataDirError(
                f"Found {len(candidates)} data pairs in {root}: {listed}. "
                f"Choose one with --prefix, e.g. --prefix {candidates[0]}"
            )
        chosen = candidates[0]

    roles = found[chosen]
    original = _single(roles["origin"], chosen, "original")
    synthetic = _single(roles["synth"], chosen, "synthetic") if roles["synth"] else None
    return DataPair(chosen, original, synthetic, root / results_dir_name(chosen))


def _not_found_message(
    root: Path, found: dict[str, dict[str, list[Path]]], need_synthetic: bool
) -> str:
    """찾지 못한 이유를 구체적으로 알려 주는 메시지."""
    origin_sfx = config.DATA_DIR_SUFFIXES["origin"]
    synth_sfx = config.DATA_DIR_SUFFIXES["synth"]
    exts = "/".join(e.lstrip(".") for e in config.DATA_DIR_EXTENSIONS)
    only_origin = sorted(p for p, r in found.items() if r["origin"] and not r["synth"])
    only_synth = sorted(p for p, r in found.items() if r["synth"] and not r["origin"])
    lines = [f"No data pair found in {root}."]
    if need_synthetic and only_origin:
        lines.append(f"  Missing synthetic file for: {', '.join(only_origin)}")
    if only_synth:
        lines.append(f"  Missing original file for: {', '.join(only_synth)}")
    lines.append(
        f"  Expected '<prefix>{origin_sfx}.<ext>' and '<prefix>{synth_sfx}.<ext>' (ext: {exts})."
    )
    return "\n".join(lines)
