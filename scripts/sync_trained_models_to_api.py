#!/usr/bin/env python3
"""
Upload each local training package to the HSM Visualiser API.

Scans ``{models_dir}/*/`` for ``package.json`` + ``model.pkl``, matches
``latin_name`` / ``activity_type`` to ``GET /models``, then ``PUT``s
``metadata`` + ``serialized_model_file``.

Auth: ``HSM_EMAIL`` / ``HSM_PASSWORD`` or ``--email`` / ``--password``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import requests

_SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_hsm_build_model_metadata",
    _SCRIPTS_DIR / "build_model_metadata_from_package.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
model_metadata_from_package = _mod.model_metadata_from_package


def _training_packages(models_dir: Path) -> Iterator[Tuple[Path, Path, Dict[str, Any]]]:
    """Yield ``(package_json, model_pkl, package_dict)`` for each valid bundle."""
    for pkg_json in sorted(models_dir.glob("*/package.json")):
        pkl = pkg_json.parent / "model.pkl"
        if not pkl.is_file():
            continue
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
        yield pkg_json, pkl, pkg


def _catalog_index(models: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    return {(row["species"], row["activity"]): row["id"] for row in models}


def _authenticated_session(base_url: str, email: str, password: str) -> requests.Session:
    s = requests.Session()
    r = s.post(
        f"{base_url}/auth/token",
        json={"email": email, "password": password, "admin_only": True},
        timeout=60,
    )
    r.raise_for_status()
    token = r.json()["id_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.environ.get("HSM_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--models-dir", type=Path, default=Path("data/sdm_models"))
    p.add_argument("--email", default=os.environ.get("HSM_EMAIL", ""))
    p.add_argument("--password", default=os.environ.get("HSM_PASSWORD", ""))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    email = args.email or os.environ.get("HSM_EMAIL", "")
    password = args.password or os.environ.get("HSM_PASSWORD", "")
    if not email or not password:
        print("Set HSM_EMAIL and HSM_PASSWORD or pass --email / --password.", file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    models_dir = args.models_dir.resolve()

    try:
        session = _authenticated_session(base, email, password)
    except requests.HTTPError as e:
        print(f"Auth failed: {e.response.status_code} {e.response.text}", file=sys.stderr)
        return 1

    r = session.get(f"{base}/models", timeout=60)
    if r.status_code != 200:
        print(f"GET /models failed HTTP {r.status_code}: {r.text}", file=sys.stderr)
        return 1

    catalog = _catalog_index(r.json())
    missing: List[Tuple[str, str, str]] = []
    failed: List[Tuple[str, str]] = []
    n_ok = 0

    for _pkg_json, pkl, pkg in _training_packages(models_dir):
        latin = pkg.get("latin_name") or ""
        activity = pkg.get("activity_type") or ""
        slug = pkg.get("model_id") or pkl.parent.name
        model_id = catalog.get((latin, activity))
        if not model_id:
            missing.append((slug, latin, activity))
            continue

        meta = model_metadata_from_package(pkg)

        if args.dry_run:
            print(f"DRY-RUN PUT {model_id} <- {slug}")
            n_ok += 1
            continue

        pr = session.put(
            f"{base}/models/{model_id}",
            data={"metadata": json.dumps(meta)},
            files={
                "serialized_model_file": (
                    "model.pkl",
                    pkl.read_bytes(),
                    "application/octet-stream",
                ),
            },
            timeout=300,
        )
        if pr.status_code != 200:
            failed.append((slug, f"HTTP {pr.status_code} {pr.text[:800]}"))
            continue

        print(f"OK {latin} — {activity} -> {model_id}")
        n_ok += 1

    for slug, latin, activity in missing:
        print(f"MISSING API ROW {slug} ({latin!r}, {activity!r})", file=sys.stderr)
    for slug, err in failed:
        print(f"FAILED {slug}: {err}", file=sys.stderr)

    print(f"Done: {n_ok} uploaded, {len(missing)} no API match, {len(failed)} errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
