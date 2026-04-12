#!/usr/bin/env python3
"""
Upload every local training package (package.json + model.pkl) to the HSM Visualiser API.

For each ``{models_dir}/{model_id}/`` with ``package.json`` and ``model.pkl``, finds the
catalog row with matching ``latin_name`` / ``activity_type`` and ``PUT``s metadata + pickle.

Credentials: ``HSM_EMAIL`` / ``HSM_PASSWORD`` or ``--email`` / ``--password``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("HSM_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--models-dir", type=Path, default=Path("data/sdm_models"))
    parser.add_argument("--email", default=os.environ.get("HSM_EMAIL", ""))
    parser.add_argument("--password", default=os.environ.get("HSM_PASSWORD", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    email = args.email or os.environ.get("HSM_EMAIL", "")
    password = args.password or os.environ.get("HSM_PASSWORD", "")
    if not email or not password:
        print("Set HSM_EMAIL and HSM_PASSWORD or pass --email / --password.", file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    r = requests.post(
        f"{base}/auth/token",
        json={"email": email, "password": password, "admin_only": True},
        timeout=60,
    )
    if r.status_code != 200:
        print(f"Auth failed HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    token = r.json()["id_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(f"{base}/models", headers=headers, timeout=60)
    if r.status_code != 200:
        print(f"GET /models failed HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    models = r.json()
    key_to_id: Dict[Tuple[str, str], str] = {}
    for row in models:
        key_to_id[(row["species"], row["activity"])] = row["id"]

    models_dir = args.models_dir.resolve()
    script = Path(__file__).resolve().parent / "build_model_metadata_from_package.py"
    ok, missing, failed = 0, [], []

    for pkg_json in sorted(models_dir.glob("*/package.json")):
        pkg_dir = pkg_json.parent
        pkl = pkg_dir / "model.pkl"
        if not pkl.is_file():
            print(f"skip (no model.pkl): {pkg_dir.name}", file=sys.stderr)
            continue
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
        latin = pkg.get("latin_name") or ""
        activity = pkg.get("activity_type") or ""
        mid = key_to_id.get((latin, activity))
        if not mid:
            missing.append((pkg_dir.name, latin, activity))
            continue

        proc = subprocess.run(
            [sys.executable, str(script), str(pkg_json)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            failed.append((pkg_dir.name, proc.stderr.strip() or proc.stdout))
            continue
        meta_str = proc.stdout.strip()

        if args.dry_run:
            print(f"DRY-RUN PUT {mid} <- {pkg_dir.name}")
            ok += 1
            continue

        try:
            meta_obj = json.loads(meta_str)
        except json.JSONDecodeError as e:
            failed.append((pkg_dir.name, str(e)))
            continue

        files = {
            "serialized_model_file": ("model.pkl", pkl.read_bytes(), "application/octet-stream"),
        }
        data = {"metadata": json.dumps(meta_obj)}
        pr = requests.put(
            f"{base}/models/{mid}",
            headers=headers,
            data=data,
            files=files,
            timeout=300,
        )
        if pr.status_code != 200:
            failed.append((pkg_dir.name, f"HTTP {pr.status_code} {pr.text[:800]}"))
            continue
        print(f"OK {latin} — {activity} -> {mid}")
        ok += 1

    for name, latin, activity in missing:
        print(f"MISSING API ROW {name} ({latin!r}, {activity!r})", file=sys.stderr)
    for name, err in failed:
        print(f"FAILED {name}: {err}", file=sys.stderr)

    print(f"Done: {ok} uploaded, {len(missing)} no API match, {len(failed)} errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
