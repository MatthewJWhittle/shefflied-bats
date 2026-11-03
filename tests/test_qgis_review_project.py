"""QGIS review project: regenerate-diff, credentials, and path allowlist (#127)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sdm.utils.qgis_review_project import (
    DEFAULT_MANIFEST_REL,
    DEFAULT_PROJECT_REL,
    assert_no_credentials,
    assert_project_paths_allowed,
    find_credential_hits,
    generate_project_text,
    load_manifest,
    version_group_title,
    write_project,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_project_matches_generator() -> None:
    """CI: regenerating from the committed manifest must match the committed .qgs."""
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    expected = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    actual = generate_project_text(manifest)
    assert actual == expected


def test_committed_project_has_no_credentials() -> None:
    text = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    assert_no_credentials(text)
    assert find_credential_hits(text) == []


def test_committed_project_paths_come_from_manifest() -> None:
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    text = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    assert_project_paths_allowed(text, manifest)


def test_deliberate_credential_fails_check() -> None:
    """Proof: injecting credential-like text is rejected."""
    text = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    poisoned = text.replace(
        "</title>",
        "</title>\n  <!-- password=secret authcfg=xyz GS_OAUTH credential: leak -->",
        1,
    )
    with pytest.raises(AssertionError, match="Credential-like content"):
        assert_no_credentials(poisoned)
    hits = find_credential_hits(poisoned)
    assert "password" in {h.lower() for h in hits} or any(
        "password" in h.lower() for h in hits
    )


def test_unknown_vsigs_path_fails_allowlist() -> None:
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    text = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    poisoned = text.replace(
        "</datasource>",
        "</datasource>\n      <datasource>/vsigs/habitat-suitability-modelling-data/not-in-manifest.tif</datasource>",
        1,
    )
    with pytest.raises(AssertionError, match="Unexpected /vsigs/"):
        assert_project_paths_allowed(poisoned, manifest)


def test_empty_compare_yields_main_only_project(tmp_path: Path) -> None:
    raw = yaml.safe_load((REPO_ROOT / DEFAULT_MANIFEST_REL).read_text(encoding="utf-8"))
    raw["versions"]["compare"] = {"label": "Unused", "models": {}}
    manifest_path = tmp_path / "manifest.yml"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    assert len(manifest.versions) == 1
    assert manifest.versions[0].version_id == "main"
    built_text = generate_project_text(manifest)
    assert "Compare:" not in built_text
    assert version_group_title(manifest.versions[0]) == "Main"
    assert 'name="Main"' in built_text
    assert_no_credentials(built_text)
    assert_project_paths_allowed(built_text, manifest)
    # Themes: Main only (no Compare / Pair).
    assert built_text.count('name="Main"') >= 1
    assert 'name="Compare"' not in built_text
    assert "Pair:" not in built_text


def test_compare_groups_and_themes_present() -> None:
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    text = generate_project_text(manifest)
    assert 'name="Main"' in text
    assert "Compare: Retrain 2 Nov" in text
    assert 'name="Compare"' in text
    assert "Pair: Myotis brandtii (Roost)" in text
    assert "/vsigs/habitat-suitability-modelling-data/predictions/versions/2026-11-02-review/2_retrain/" in text
    assert "myotis_brandtii_roost__suitability.tif" in text
    assert "myotis_brandtii_roost__cutoff_p10.tif" in text
    # Google Satellite must stay out.
    assert "google.cn" not in text.lower()
    assert "Google Satellite" not in text


def test_write_project_round_trip(tmp_path: Path) -> None:
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    out = tmp_path / "out.qgs"
    write_project(manifest, repo_root=tmp_path, output_path=out)
    assert out.read_text(encoding="utf-8") == generate_project_text(manifest)
