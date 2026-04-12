#!/usr/bin/env python3
"""CLI: print one-line ``ModelMetadata`` JSON from a training ``package.json``.

Run from project root (after ``cd`` there) so ``sdm`` is importable:

  uv run python scripts/build_model_metadata_from_package.py path/to/package.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sdm.utils.hsm_metadata import model_metadata_from_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_json",
        type=Path,
        help="Path to package.json next to model.pkl",
    )
    args = parser.parse_args()
    pkg = json.loads(args.package_json.read_text(encoding="utf-8"))
    json.dump(model_metadata_from_package(pkg), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
