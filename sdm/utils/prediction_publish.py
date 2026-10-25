"""Versioned prediction map publishing for QGIS browsing.

Layout (under a publish root configured via ``--publish-root``,
``PREDICTION_PUBLISH_ROOT``, or ``prediction_publish.root`` in ``config.yml``)::

    <model_id>/<YYYYMMDD>_<sha7>_cfg<config_hash>/
        prediction__<genus>_<species>_<behaviour>__<version>__suitability.tif
        prediction__…__threshold_<rule>.tif   # omitted when no cut-off
        prediction__…__run.json

File names are ordered general → specific so QGIS layer titles stay identifiable.
Folder path is unchanged. Publishing is opt-in. Local predict outputs stay under
the normal predictions directory unless a publish root is requested. Version
folders are never overwritten. The library has no hard-coded project bucket or
species map; those live in project config.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from sdm.commands.modelling.utils import clean_string
from sdm.models.core.prediction import (
    copy_prediction_as_cog,
    write_threshold_binary_raster,
)
from sdm.models.core.threshold import (
    POOLED_OOF_PERCENTILE_RULE,
    RUN_THRESHOLDS_JSON,
    _json_safe,
)
from sdm.utils.fs import (
    filesystem_for_uri,
    join_uri_path,
    put_local_tree,
    write_text_idempotent,
)
from sdm.utils.io import load_config
from sdm.utils.rebuild_provenance import git_commit_sha

logger = logging.getLogger(__name__)

PREDICTION_PUBLISH_ROOT_ENV = "PREDICTION_PUBLISH_ROOT"
CONFIG_HASH_LENGTH = 8
SUITABILITY_KIND = "suitability"
RUN_KIND = "run"
_MISSING_PUBLISH_ROOT_MSG = (
    "Prediction publish root is not configured. Pass --publish-root, "
    "set PREDICTION_PUBLISH_ROOT, or set prediction_publish.root in config.yml."
)


def _prediction_publish_section(
    config: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Return the ``prediction_publish`` mapping from *config* or ``config.yml``."""
    if config is None:
        config = load_config()
    section = config.get("prediction_publish") if isinstance(config, Mapping) else None
    if isinstance(section, Mapping):
        return section
    return {}


def load_species_binomial_by_key(
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    """Load epithet → ``genus_species`` map from project ``prediction_publish`` config."""
    section = _prediction_publish_section(config)
    raw = section.get("species_binomial_by_key") or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def resolve_prediction_publish_root(
    cli_root: Optional[Union[str, Path]] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve publish root: CLI, else env, else ``prediction_publish.root``.

    Raises:
        ValueError: when no root is set (library has no hard-coded default).
    """
    if cli_root is not None and str(cli_root).strip():
        return str(cli_root).rstrip("/")
    env_root = os.environ.get(PREDICTION_PUBLISH_ROOT_ENV, "").strip()
    if env_root:
        return env_root.rstrip("/")
    section = _prediction_publish_section(config)
    config_root = section.get("root")
    if config_root is not None and str(config_root).strip():
        return str(config_root).rstrip("/")
    raise ValueError(_MISSING_PUBLISH_ROOT_MSG)


def git_working_tree_dirty(*, cwd: Optional[Path] = None) -> bool:
    """Return True when the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool((result.stdout or "").strip())


def stable_config_hash(config: Mapping[str, Any], *, length: int = CONFIG_HASH_LENGTH) -> str:
    """Return a short hex digest of a JSON-stable config payload."""
    payload = json.dumps(
        _json_safe(dict(config)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


def build_version_folder_name(
    *,
    created_at: Optional[datetime] = None,
    git_sha: Optional[str] = None,
    config_hash: str,
) -> str:
    """Build ``<YYYYMMDD>_<sha7>_cfg<hash>`` for a prediction version folder."""
    moment = created_at or datetime.now(timezone.utc)
    date_stamp = moment.strftime("%Y%m%d")
    sha = (git_sha or "").strip()
    sha7 = sha[:7] if sha else "unknown"
    return f"{date_stamp}_{sha7}_cfg{config_hash}"


def prediction_version_uri(
    publish_root: str,
    model_id: str,
    version_folder: str,
) -> str:
    return join_uri_path(publish_root, model_id, version_folder)


def resolve_binomial_slug(
    latin_name: str,
    *,
    species_binomial_by_key: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve a latin name or epithet to ``genus_species`` via the project map.

    Unknown full Latin binomials (two or more ``_``-separated parts) pass through
    unchanged so model ids that are already binomial never get a doubled genus.
    """
    mapping = (
        dict(species_binomial_by_key)
        if species_binomial_by_key is not None
        else load_species_binomial_by_key()
    )
    binomial_slugs = frozenset(mapping.values())
    key = clean_string(latin_name)
    if key in mapping:
        return mapping[key]
    if key in binomial_slugs:
        return key
    parts = key.split("_")
    if len(parts) >= 2:
        # Full binomial already (including species not listed in the short map).
        return key
    if parts and parts[-1] in mapping:
        return mapping[parts[-1]]
    raise ValueError(
        f"Unknown species for prediction publish naming: {latin_name!r}; "
        f"known keys: {sorted(mapping)}"
    )


def normalize_behaviour(activity_type: str) -> str:
    """Return ``roost`` or ``in_flight`` for published file names."""
    cleaned = clean_string(activity_type)
    if cleaned in {"roost"}:
        return "roost"
    if cleaned in {"in_flight", "inflight", "foraging", "in-flight"}:
        return "in_flight"
    raise ValueError(
        f"Unknown behaviour/activity for prediction publish naming: {activity_type!r} "
        "(expected roost or in_flight)"
    )


def publish_name_slug(
    *,
    latin_name: Optional[str] = None,
    activity_type: Optional[str] = None,
    model_id: Optional[str] = None,
    species_binomial_by_key: Optional[Mapping[str, str]] = None,
) -> str:
    """Build ``<genus>_<species>_<behaviour>`` used inside published file names."""
    if latin_name and activity_type:
        binomial = resolve_binomial_slug(
            latin_name, species_binomial_by_key=species_binomial_by_key
        )
        behaviour = normalize_behaviour(activity_type)
        return f"{binomial}_{behaviour}"

    if not model_id:
        raise ValueError("latin_name+activity_type or model_id required for publish naming")

    mid = clean_string(model_id)
    for behaviour in ("in_flight", "roost"):
        suffix = f"_{behaviour}"
        if mid.endswith(suffix):
            species_part = mid[: -len(suffix)]
            binomial = resolve_binomial_slug(
                species_part, species_binomial_by_key=species_binomial_by_key
            )
            return f"{binomial}_{behaviour}"
    raise ValueError(
        f"Cannot derive publish slug from model_id={model_id!r}; "
        "expected …_roost or …_in_flight"
    )


def published_artifact_filename(
    publish_slug: str,
    version_folder: str,
    kind: str,
    *,
    ext: str,
) -> str:
    """``prediction__<slug>__<version>__<kind>.<ext>`` for QGIS layer names."""
    return (
        f"prediction__{publish_slug}__{version_folder}__{kind}.{ext.lstrip('.')}"
    )


def threshold_kind_for_rule(rule: str) -> str:
    """Map a cut-off rule id to the published threshold file kind."""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in rule)
    return f"threshold_{safe}"


def version_folder_exists(version_uri: str) -> bool:
    """True when a version folder already has published artefacts."""
    fs, path = filesystem_for_uri(version_uri.rstrip("/"))
    path = path.rstrip("/")
    try:
        if fs.exists(path) and getattr(fs, "isdir", lambda _p: False)(path):
            listing = fs.ls(path, detail=False)
            return bool(listing)
    except FileNotFoundError:
        return False
    return False


def threshold_raster_filename(
    publish_slug: str,
    version_folder: str,
    rule: str,
) -> str:
    """Published threshold GeoTIFF name (includes species, behaviour, version)."""
    return published_artifact_filename(
        publish_slug,
        version_folder,
        threshold_kind_for_rule(rule),
        ext="tif",
    )


def load_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return None


def load_model_threshold(
    models_dir: Path,
    model_id: str,
) -> dict[str, Any]:
    """Load cut-off metadata for *model_id* from the training run folder."""
    package_threshold = load_json_if_exists(
        Path(models_dir) / model_id / RUN_THRESHOLDS_JSON
    )
    if package_threshold is not None:
        return package_threshold

    run_threshold = load_json_if_exists(Path(models_dir) / RUN_THRESHOLDS_JSON)
    if run_threshold is None:
        return {
            "rule": POOLED_OOF_PERCENTILE_RULE,
            "threshold": None,
            "model_limited": False,
            "no_cutoff_reason": "thresholds_missing",
        }

    models = run_threshold.get("models") or []
    for row in models:
        if isinstance(row, Mapping) and row.get("model_id") == model_id:
            return dict(row)
    thresholds_map = run_threshold.get("thresholds") or {}
    return {
        "rule": run_threshold.get("rule", POOLED_OOF_PERCENTILE_RULE),
        "threshold": thresholds_map.get(model_id),
        "percentile": run_threshold.get("percentile"),
        "model_limited": thresholds_map.get(model_id) is None
        and bool(run_threshold.get("thresholds")),
        "no_cutoff_reason": None,
    }


def environmental_layer_versions(
    band_names: Sequence[str],
    *,
    rebuild_metrics_path: Optional[Path] = None,
) -> dict[str, Optional[str]]:
    """Map each EV band name to a version string when rebuild metrics exist."""
    versions: dict[str, Optional[str]] = {name: None for name in band_names}
    if rebuild_metrics_path is None or not rebuild_metrics_path.is_file():
        return versions

    metrics = load_json_if_exists(rebuild_metrics_path)
    if not metrics:
        return versions

    provenance = metrics.get("provenance") or {}
    sources = provenance.get("sources") or {}
    # Best-effort: match band name prefixes to provenance source keys / products.
    for band in band_names:
        matched: Optional[str] = None
        for source_key, source_meta in sources.items():
            if not isinstance(source_meta, Mapping):
                continue
            if band.startswith(source_key) or source_key in band:
                matched = (
                    source_meta.get("version")
                    or source_meta.get("product")
                    or source_meta.get("dataset_name")
                )
                if matched is not None:
                    matched = str(matched)
                break
        versions[band] = matched
    return versions


def build_run_manifest(
    *,
    model_id: str,
    version_folder: str,
    publish_slug: str,
    config: Mapping[str, Any],
    config_hash: str,
    git_sha: Optional[str],
    git_dirty: bool,
    cutoff_rule: Optional[str],
    cutoff_value: Optional[float],
    cutoff_meta: Optional[Mapping[str, Any]] = None,
    cv_metrics: Optional[Mapping[str, Any]] = None,
    environmental_layers: Optional[Mapping[str, Optional[str]]] = None,
    mlflow_run_url: Optional[str] = None,
    label: Optional[str] = None,
    created_at: Optional[datetime] = None,
    files: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, Any]:
    """Assemble published run metadata (written as ``…__run.json``)."""
    moment = created_at or datetime.now(timezone.utc)
    cutoff: dict[str, Any] = {
        "rule": cutoff_rule,
        "value": None if cutoff_value is None else float(cutoff_value),
    }
    if cutoff_meta:
        for key in (
            "percentile",
            "heldout_caught",
            "map_share",
            "flag",
            "model_limited",
            "no_cutoff_reason",
        ):
            if key in cutoff_meta:
                cutoff[key] = cutoff_meta[key]

    return _json_safe(
        {
            "model_id": model_id,
            "publish_slug": publish_slug,
            "version": version_folder,
            "git_sha": git_sha,
            "git_dirty": bool(git_dirty),
            "config": dict(config),
            "config_hash": config_hash,
            "environmental_layer_versions": dict(environmental_layers or {}),
            "cutoff": cutoff,
            "cv_metrics": dict(cv_metrics or {}),
            "mlflow_run_url": mlflow_run_url,
            "label": label,
            "files": dict(files or {}),
            "created_at": moment.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
    )


def _publish_local_tree(local_dir: Path, target_uri: str) -> None:
    """Copy a staged tree to *target_uri* (local path or remote URI)."""
    fs, remote_path = filesystem_for_uri(target_uri.rstrip("/"))
    protocol = fs.protocol
    if isinstance(protocol, (tuple, list)):
        protocol = protocol[0]
    if protocol == "file":
        dest = Path(remote_path)
        dest.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(local_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(local_dir)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, out)
        return
    fs.makedirs(remote_path, exist_ok=True)
    put_local_tree(fs, local_dir, remote_path)


def publish_model_prediction_version(
    *,
    model_id: str,
    prediction_raster: Path,
    publish_root: str,
    models_dir: Path,
    config: Mapping[str, Any],
    label: Optional[str] = None,
    mlflow_run_url: Optional[str] = None,
    rebuild_metrics_path: Optional[Path] = None,
    band_names: Optional[Sequence[str]] = None,
    created_at: Optional[datetime] = None,
    git_sha: Optional[str] = None,
    git_dirty: Optional[bool] = None,
    write_cog: bool = True,
    species_binomial_by_key: Optional[Mapping[str, str]] = None,
) -> str:
    """Publish one model's prediction maps into a new version folder.

    Raises:
        FileExistsError: if the computed version folder already exists.
        FileNotFoundError: if *prediction_raster* is missing.
    """
    prediction_raster = Path(prediction_raster)
    if not prediction_raster.is_file():
        raise FileNotFoundError(f"Prediction raster not found: {prediction_raster}")

    package = load_json_if_exists(Path(models_dir) / model_id / "package.json") or {}
    threshold_meta = load_model_threshold(Path(models_dir), model_id)
    rule = str(threshold_meta.get("rule") or POOLED_OOF_PERCENTILE_RULE)
    cutoff_value = threshold_meta.get("threshold")
    if cutoff_value is not None:
        cutoff_value = float(cutoff_value)

    full_config = {
        "model_id": model_id,
        "package": package,
        "threshold": threshold_meta,
        "prediction": dict(config),
    }
    config_hash = stable_config_hash(full_config)
    sha = git_sha if git_sha is not None else git_commit_sha()
    dirty = git_working_tree_dirty() if git_dirty is None else bool(git_dirty)
    moment = created_at or datetime.now(timezone.utc)
    version_folder = build_version_folder_name(
        created_at=moment,
        git_sha=sha,
        config_hash=config_hash,
    )
    version_uri = prediction_version_uri(publish_root, model_id, version_folder)

    if version_folder_exists(version_uri):
        raise FileExistsError(
            "Cannot publish prediction version "
            f"(refusing to overwrite {version_uri})"
        )

    layer_names = list(band_names or package.get("feature_names") or [])
    env_versions = environmental_layer_versions(
        layer_names,
        rebuild_metrics_path=rebuild_metrics_path,
    )
    cv_metrics = package.get("metrics") if isinstance(package.get("metrics"), dict) else {}
    if mlflow_run_url is None:
        mlflow_run_url = package.get("mlflow_run_url") or (
            (package.get("artifacts") or {}).get("mlflow_run_url")
        )

    latin_name = package.get("latin_name")
    activity_type = package.get("activity_type")
    name_slug = publish_name_slug(
        latin_name=str(latin_name) if latin_name else None,
        activity_type=str(activity_type) if activity_type else None,
        model_id=str(package.get("model_id") or model_id),
        species_binomial_by_key=species_binomial_by_key,
    )

    suitability_name = published_artifact_filename(
        name_slug, version_folder, SUITABILITY_KIND, ext="tif"
    )
    run_name = published_artifact_filename(
        name_slug, version_folder, RUN_KIND, ext="json"
    )
    threshold_name: Optional[str] = None
    write_threshold = cutoff_value is not None and not threshold_meta.get(
        "model_limited"
    )
    if write_threshold:
        threshold_name = threshold_raster_filename(name_slug, version_folder, rule)

    files = {
        "suitability": suitability_name,
        "threshold": threshold_name,
        "run": run_name,
    }
    run_manifest = build_run_manifest(
        model_id=model_id,
        version_folder=version_folder,
        publish_slug=name_slug,
        config=full_config,
        config_hash=config_hash,
        git_sha=sha,
        git_dirty=dirty,
        cutoff_rule=rule,
        cutoff_value=cutoff_value,
        cutoff_meta=threshold_meta,
        cv_metrics=cv_metrics,
        environmental_layers=env_versions,
        mlflow_run_url=mlflow_run_url,
        label=label,
        created_at=moment,
        files=files,
    )

    with tempfile.TemporaryDirectory(prefix="sdm-pred-pub-") as tmp:
        staging = Path(tmp)
        dest_suitability = staging / suitability_name
        if write_cog:
            copy_prediction_as_cog(prediction_raster, dest_suitability)
        else:
            shutil.copy2(prediction_raster, dest_suitability)

        if write_threshold and threshold_name is not None:
            write_threshold_binary_raster(
                prediction_raster,
                staging / threshold_name,
                cutoff=float(cutoff_value),
                as_cog=write_cog,
            )

        write_text_idempotent(
            staging / run_name,
            json.dumps(run_manifest, indent=2, allow_nan=False) + "\n",
        )

        # Re-check immediately before upload to reduce race windows.
        if version_folder_exists(version_uri):
            raise FileExistsError(
                "Cannot publish prediction version "
                f"(refusing to overwrite {version_uri})"
            )
        _publish_local_tree(staging, version_uri)

    logger.info("Published prediction version for %s → %s", model_id, version_uri)
    return version_uri


def publish_predictions_from_output_dir(
    *,
    output_dir: Path,
    models_dir: Path,
    publish_root: str,
    model_ids: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
    prediction_crs: Optional[str] = None,
    write_cog: bool = True,
    rebuild_metrics_path: Optional[Path] = None,
    species_binomial_by_key: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Publish each ``prediction_<model_id>.tif`` under *output_dir*."""
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    published: list[str] = []

    prediction_paths = sorted(output_dir.glob("prediction_*.tif"))
    if model_ids is not None:
        wanted = {mid.lower() for mid in model_ids}
        prediction_paths = [
            path
            for path in prediction_paths
            if path.stem[len("prediction_"):].lower() in wanted
        ]

    if not prediction_paths:
        raise FileNotFoundError(
            f"No per-model prediction_*.tif files found under {output_dir}"
        )

    predict_config = {
        "prediction_crs": prediction_crs,
        "write_cog": write_cog,
        "local_output_dir": str(output_dir),
    }

    for path in prediction_paths:
        model_id = path.stem[len("prediction_"):]
        uri = publish_model_prediction_version(
            model_id=model_id,
            prediction_raster=path,
            publish_root=publish_root,
            models_dir=models_dir,
            config=predict_config,
            label=label,
            rebuild_metrics_path=rebuild_metrics_path,
            write_cog=write_cog,
            species_binomial_by_key=species_binomial_by_key,
        )
        published.append(uri)
    return published
