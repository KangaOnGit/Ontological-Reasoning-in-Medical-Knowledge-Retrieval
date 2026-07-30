from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def write_submission_zip(
    files: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
) -> Path:
    """
    Write the competition submission ZIP.

    Parameters
    ----------
    files: Mapping from input file stem to its prediction records.
        {
            "1": [...],
            "2": [...],
            "15": [...],
        }

    Produces
        submission.zip
        |-- 1.json
        |-- 2.json
        |-- 15.json
        L ...
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stems = sorted(files, key=int)
    except ValueError:
        stems = sorted(files)

    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for stem in stems:
            zf.writestr(
                f"{stem}.json",
                json.dumps(
                    files[stem],
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    return output_path


def write_json(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write prediction records for one input file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path