"""QGIS review project tooling: regenerate-diff, credentials, path allowlist (#127).

Lives outside ``sdm`` — imports ``qgis.review.generate`` only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from qgis.review.generate import (
    BASEMAP_CRS,
    DEFAULT_MANIFEST_REL,
    DEFAULT_PROJECT_REL,
    _stable_layer_id,
    assert_no_credentials,
    assert_project_paths_allowed,
    build_project_xml,
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
    assert built_text.count('name="Main"') >= 1
    assert 'name="Compare"' not in built_text
    assert "Pair:" not in built_text
    assert "— Compare" not in built_text


def test_compare_groups_and_themes_present() -> None:
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    text = generate_project_text(manifest)
    assert 'name="Main"' in text
    assert "Compare: Retrain 2 Nov" in text
    assert 'name="Compare"' in text
    assert "Pair:" not in text
    assert 'name="Myotis brandtii Roost — Main"' in text
    assert 'name="Myotis brandtii Roost — Compare"' in text
    assert "/vsigs/habitat-suitability-modelling-data/predictions/versions/2026-11-02-review/2_retrain/" in text
    assert "myotis_brandtii_roost__suitability.tif" in text
    assert "myotis_brandtii_roost__cutoff_p10.tif" in text
    assert "google.cn" not in text.lower()
    assert "Google Satellite" not in text
    assert "Esri World Imagery" in text
    assert "server.arcgisonline.com/ArcGIS/rest/services/World_Imagery" in text
    assert 'opacity="0.8"' in text
    assert 'color="#440154"' in text
    assert 'color="#fde725"' in text


def test_layer_names_include_species_and_version() -> None:
    """Suitability layers must not share a generic name across versions."""
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    text = generate_project_text(manifest)
    assert "<layername>Myotis brandtii Roost — Main</layername>" in text
    assert "<layername>Myotis brandtii Roost — Compare</layername>" in text
    assert "modelled suitability (0–1)" not in text


def test_per_model_themes_show_one_version_only() -> None:
    """Per-model themes must not stack Main and Compare suitability."""
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    text = build_project_xml(manifest).xml
    main_suit = _stable_layer_id("main", "myotis_brandtii_roost", "suitability")
    compare_suit = _stable_layer_id(
        "compare", "myotis_brandtii_roost", "suitability"
    )

    def _preset_body(theme_name: str) -> str:
        marker = f'<visibility-preset name="{theme_name}"'
        assert marker in text
        return text.split(marker, 1)[1].split("</visibility-preset>", 1)[0]

    def _visible(theme_body: str, layer_id: str) -> str:
        marker = f'id="{layer_id}"'
        assert marker in theme_body
        attrs = theme_body.split(marker, 1)[1].split("/>", 1)[0]
        if 'visible="1"' in attrs:
            return "1"
        if 'visible="0"' in attrs:
            return "0"
        raise AssertionError(f"missing visible attr for {layer_id}")

    main_theme = _preset_body("Myotis brandtii Roost — Main")
    compare_theme = _preset_body("Myotis brandtii Roost — Compare")
    assert _visible(main_theme, main_suit) == "1"
    assert _visible(main_theme, compare_suit) == "0"
    assert _visible(compare_theme, compare_suit) == "1"
    assert _visible(compare_theme, main_suit) == "0"


def test_write_project_round_trip(tmp_path: Path) -> None:
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    out = tmp_path / "out.qgs"
    write_project(manifest, repo_root=tmp_path, output_path=out)
    assert out.read_text(encoding="utf-8") == generate_project_text(manifest)


def test_generator_is_outside_sdm() -> None:
    """QGIS review tooling must not live under the modelling library."""
    assert not (REPO_ROOT / "sdm" / "utils" / "qgis_review_project.py").exists()
    assert (REPO_ROOT / "qgis" / "review" / "generate.py").is_file()
    import sdm

    sdm_root = Path(sdm.__file__).resolve().parent
    generate_path = (REPO_ROOT / "qgis" / "review" / "generate.py").resolve()
    assert sdm_root not in generate_path.parents


def test_non_xyz_layer_srs_matches_manifest_crs() -> None:
    """Every prediction/cutoff maplayer authid must equal manifest.crs."""
    manifest = load_manifest(REPO_ROOT / DEFAULT_MANIFEST_REL)
    assert manifest.crs == "EPSG:27700"
    text = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    # Split into maplayer blocks; skip XYZ basemaps (provider wms / type=xyz).
    layers = re.findall(r"<maplayer\b[\s\S]*?</maplayer>", text)
    assert layers
    checked = 0
    for block in layers:
        if "type=xyz" in block or "<provider>wms</provider>" in block:
            auth = re.search(r"<authid>([^<]+)</authid>", block)
            assert auth is not None
            assert auth.group(1) == BASEMAP_CRS
            continue
        auth = re.search(r"<authid>([^<]+)</authid>", block)
        assert auth is not None, block[:200]
        assert auth.group(1) == manifest.crs
        checked += 1
    assert checked >= 1
    # Project CRS follows the same manifest value.
    project_crs = re.search(
        r"<projectCrs>\s*<spatialrefsys>[\s\S]*?<authid>([^<]+)</authid>",
        text,
    )
    assert project_crs is not None
    assert project_crs.group(1) == manifest.crs


def test_projections_enabled_property_is_one() -> None:
    """QGIS 4.x needs SpatialRefSys/ProjectionsEnabled=1 for a live project CRS."""
    text = (REPO_ROOT / DEFAULT_PROJECT_REL).read_text(encoding="utf-8")
    match = re.search(
        r"<SpatialRefSys>\s*<ProjectionsEnabled type=\"int\">(\d+)</ProjectionsEnabled>\s*</SpatialRefSys>",
        text,
    )
    assert match is not None
    assert match.group(1) == "1"
