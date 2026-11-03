#!/usr/bin/env python3
"""Regenerate ``qgis/review/hsm-review.qgs`` from ``qgis/review/manifest.yml``.

Usage (from repo root)::

    uv run python scripts/generate_qgis_review_project.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdm.utils.qgis_review_project import (  # noqa: E402
    DEFAULT_MANIFEST_REL,
    load_manifest,
    write_project,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / DEFAULT_MANIFEST_REL,
        help="Path to the review manifest YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .qgs path (default: project path from the manifest)",
    )
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    out = write_project(manifest, repo_root=REPO_ROOT, output_path=args.output)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
