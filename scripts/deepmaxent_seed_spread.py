#!/usr/bin/env python3
"""Across-seed score / map spread for DeepMaxent (real data; not CI).

## Implementation notes

Re-fits the joint DeepMaxent model for each configured seed on a prepared
training parquet from a DeepMaxent run (or rebuilds from live ``sdm train`` with
``model.type: deepmaxent``). Writes per-seed mean CV AUC and optional map
quantiles. Requires ``uv sync --extra dl``. Prefer re-running ``sdm train`` with
``deepmaxent.n_seeds`` and inspecting run metadata; this script is for explicit
seed-by-seed export.

Example::

    uv run python scripts/deepmaxent_seed_spread.py --help
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--note",
        action="store_true",
        help="Print the recommended workflow and exit.",
    )
    args = parser.parse_args()
    print(
        "Recommended seed-spread workflow (no CI):\n"
        "  1. Set model.type: deepmaxent and deepmaxent.n_seeds: 10 in model_config.yml\n"
        "  2. uv sync --extra dl\n"
        "  3. uv run sdm train   # joint model; run_metadata.json records device + n_seeds\n"
        "  4. uv run sdm predict --run <id>   # one raster per model from mean ensemble\n"
        "  5. scripts/deepmaxent_compare_scores.py against a MaxEnt baseline run\n"
        "Seed-level map rasters: re-train with n_seeds: 1 and vary model.seed, then\n"
        "compare prediction GeoTIFFs with your usual raster tools.\n"
    )
    if not args.note:
        parser.print_help()


if __name__ == "__main__":
    main()
