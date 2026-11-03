"""Generate the in-repo QGIS review project from a YAML manifest.

Project tooling (not part of the ``sdm`` modelling library/CLI). Prediction
layer paths use GDAL ``/vsigs/`` so QGIS loads versioned maps from the bucket
without downloading. Naming for the ``main`` batch follows the
prediction-version layout from prediction publish (#104). No credentials are
written into the project file.

Run from the repo root::

    uv run python -m qgis.review.generate
"""

from __future__ import annotations

import argparse
import html
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from xml.etree.ElementTree import Element, SubElement, tostring

import yaml


def uri_for_vsigs(uri_or_path: str | Path) -> str:
    """Map ``gs://`` URIs to GDAL ``/vsigs/`` paths (local helper; no sdm import)."""
    ref = str(uri_or_path)
    if ref.startswith("gs://"):
        return f"/vsigs/{ref[len('gs://'):]}"
    return ref


LAYER_ID_NAMESPACE = uuid.UUID("6b2f0c3e-9a41-4d7a-8f12-127000000001")
DEFAULT_MANIFEST_REL = Path("qgis/review/manifest.yml")
DEFAULT_PROJECT_REL = Path("qgis/review/hsm-review.qgs")
PROJECT_TITLE = "Sheffield bats — review"
DEFAULT_RASTER_CRS = "EPSG:27700"
BASEMAP_CRS = "EPSG:3857"

# Shared 0–1 suitability ramp (matches Matt's desktop reference styling).
SUITABILITY_RAMP_ITEMS: tuple[tuple[float, str, str], ...] = (
    (0.0, "#f7fbff", "0"),
    (0.125, "#deebf7", "0.125"),
    (0.25, "#c6dbef", "0.25"),
    (0.375, "#9ecae1", "0.375"),
    (0.5, "#6baed6", "0.5"),
    (0.625, "#4292c6", "0.625"),
    (0.75, "#2171b5", "0.75"),
    (0.875, "#08519c", "0.875"),
    (1.0, "#08306b", "1"),
)

BASEMAP_SPECS: tuple[tuple[str, str], ...] = (
    (
        "OpenStreetMap",
        "crs=EPSG:3857&format&type=xyz&url="
        "https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png"
        "&zmax=19&zmin=0",
    ),
    (
        "Carto",
        "crs=EPSG:3857&format&type=xyz&url="
        "https://a.basemaps.cartocdn.com/light_all/%7Bz%7D/%7Bx%7D/%7By%7D.png"
        "&zmax=18&zmin=0&http-header:referer=",
    ),
)

# Study-area-ish default extent in EPSG:27700 (from the desktop reference).
DEFAULT_EXTENT = (
    406745.1590511440881528,
    368708.34456809132825583,
    554321.31790885585360229,
    479441.74501086655072868,
)

CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"password",
        r"\btoken\b",
        r"authcfg",
        r"GS_OAUTH",
        r"credential:",
        r"credentials",
    )
)

FORBIDDEN_BASEMAP_SNIPPETS: tuple[str, ...] = (
    "google.cn",
    "maps.google",
    "Google Satellite",
)


@dataclass(frozen=True)
class ModelLayerSpec:
    model_id: str
    run_folder: str
    prediction_file: str
    cutoff_file: Optional[str] = None


@dataclass(frozen=True)
class VersionSpec:
    version_id: str
    label: str
    models: tuple[ModelLayerSpec, ...]


@dataclass(frozen=True)
class ReviewManifest:
    versions: tuple[VersionSpec, ...]
    title: str = PROJECT_TITLE
    project_relpath: Path = DEFAULT_PROJECT_REL
    # Authid for prediction/cutoff rasters and the project (e.g. EPSG:27700).
    crs: str = DEFAULT_RASTER_CRS


@dataclass
class BuiltLayer:
    layer_id: str
    name: str
    datasource: str
    provider: str
    kind: str  # suitability | cutoff | basemap
    version_id: Optional[str]
    model_id: Optional[str]
    species_group: Optional[str]
    checked: bool = False


@dataclass
class BuiltProject:
    xml: str
    layers: list[BuiltLayer] = field(default_factory=list)
    theme_names: list[str] = field(default_factory=list)
    group_paths: list[str] = field(default_factory=list)


def _stable_layer_id(*parts: str) -> str:
    key = "|".join(parts)
    digest = uuid.uuid5(LAYER_ID_NAMESPACE, key).hex
    return f"l_{digest}"


def species_group_name(model_id: str) -> str:
    """``myotis_brandtii_roost`` → ``Myotis brandtii``."""
    cleaned = model_id.strip().lower()
    for suffix in ("_in_flight", "_roost"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    parts = cleaned.split("_")
    if len(parts) < 2:
        raise ValueError(f"model_id must be genus_species_behaviour; got {model_id!r}")
    genus, species = parts[0], parts[1]
    return f"{genus.capitalize()} {species}"


def behaviour_label(model_id: str) -> str:
    cleaned = model_id.strip().lower()
    if cleaned.endswith("_in_flight"):
        return "In flight"
    if cleaned.endswith("_roost"):
        return "Roost"
    raise ValueError(f"model_id must end with _roost or _in_flight; got {model_id!r}")


def model_display_name(model_id: str) -> str:
    """``myotis_brandtii_roost`` → ``Myotis brandtii Roost``."""
    return f"{species_group_name(model_id)} {behaviour_label(model_id)}"


def version_short_tag(version_id: str) -> str:
    """Map version id to the short tag used in layer / theme names."""
    if version_id == "main":
        return "Main"
    if version_id == "compare":
        return "Compare"
    raise ValueError(f"Unknown version_id for layer naming: {version_id!r}")


def suitability_layer_name(model_id: str, version_id: str) -> str:
    """Distinct per-version name, e.g. ``Myotis brandtii Roost — Main``."""
    return f"{model_display_name(model_id)} — {version_short_tag(version_id)}"


def cutoff_layer_name(model_id: str, version_id: str) -> str:
    return (
        f"{model_display_name(model_id)} cut-off (p10) — "
        f"{version_short_tag(version_id)}"
    )


def version_group_title(version: VersionSpec) -> str:
    if version.version_id == "main":
        return "Main"
    return f"Compare: {version.label}"


def gs_join(run_folder: str, filename: str) -> str:
    root = run_folder.rstrip("/")
    name = filename.lstrip("/")
    if not root.startswith("gs://"):
        raise ValueError(f"run_folder must be a gs:// URI; got {run_folder!r}")
    return f"{root}/{name}"


def vsigs_path(run_folder: str, filename: str) -> str:
    return uri_for_vsigs(gs_join(run_folder, filename))


def _parse_model_entry(model_id: str, raw: Mapping[str, Any]) -> ModelLayerSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"models.{model_id} must be a mapping")
    run_folder = str(raw.get("run_folder", "")).strip()
    prediction_file = str(raw.get("prediction_file", "")).strip()
    if not run_folder or not prediction_file:
        raise ValueError(
            f"models.{model_id} requires run_folder and prediction_file"
        )
    cutoff_raw = raw.get("cutoff_file")
    cutoff_file = str(cutoff_raw).strip() if cutoff_raw else None
    if cutoff_file == "":
        cutoff_file = None
    return ModelLayerSpec(
        model_id=model_id,
        run_folder=run_folder,
        prediction_file=prediction_file,
        cutoff_file=cutoff_file,
    )


def _parse_version(version_id: str, raw: Any) -> Optional[VersionSpec]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"versions.{version_id} must be a mapping or null")
    models_raw = raw.get("models")
    if models_raw is None or models_raw == {}:
        return None
    if not isinstance(models_raw, Mapping):
        raise ValueError(f"versions.{version_id}.models must be a mapping")
    label = str(raw.get("label", "")).strip()
    if not label:
        raise ValueError(f"versions.{version_id} requires a short label")
    models = tuple(
        _parse_model_entry(model_id, entry)
        for model_id, entry in models_raw.items()
    )
    if not models:
        return None
    return VersionSpec(version_id=version_id, label=label, models=models)


def load_manifest(path: Path | str) -> ReviewManifest:
    """Load and validate a review-project manifest."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be a mapping")
    versions_raw = payload.get("versions")
    if not isinstance(versions_raw, Mapping):
        raise ValueError("manifest.versions must be a mapping")
    if "main" not in versions_raw:
        raise ValueError("manifest.versions.main is required")
    main = _parse_version("main", versions_raw["main"])
    if main is None:
        raise ValueError("manifest.versions.main must list at least one model")
    versions: list[VersionSpec] = [main]
    if "compare" in versions_raw:
        compare = _parse_version("compare", versions_raw["compare"])
        if compare is not None:
            versions.append(compare)
    title = str(payload.get("title") or PROJECT_TITLE)
    project_rel = Path(str(payload.get("project") or DEFAULT_PROJECT_REL))
    crs_raw = payload.get("crs", DEFAULT_RASTER_CRS)
    crs = str(crs_raw).strip() if crs_raw is not None else DEFAULT_RASTER_CRS
    if not crs:
        crs = DEFAULT_RASTER_CRS
    if not re.fullmatch(r"EPSG:\d+", crs, flags=re.IGNORECASE):
        raise ValueError(f"manifest.crs must look like EPSG:27700; got {crs!r}")
    crs = f"EPSG:{crs.split(':', 1)[1]}"
    return ReviewManifest(
        versions=tuple(versions),
        title=title,
        project_relpath=project_rel,
        crs=crs,
    )


def manifest_vsigs_paths(manifest: ReviewManifest) -> set[str]:
    """Return every prediction / cutoff ``/vsigs/`` path declared in the manifest."""
    paths: set[str] = set()
    for version in manifest.versions:
        for model in version.models:
            paths.add(vsigs_path(model.run_folder, model.prediction_file))
            if model.cutoff_file:
                paths.add(vsigs_path(model.run_folder, model.cutoff_file))
    return paths


def fixed_basemap_datasources() -> set[str]:
    return {datasource for _, datasource in BASEMAP_SPECS}


def _sub(parent: Element, tag: str, text: Optional[str] = None, **attrs: str) -> Element:
    el = SubElement(parent, tag, {k: v for k, v in attrs.items()})
    if text is not None:
        el.text = text
    return el


def _extent_el(parent: Element, tag: str, extent: Sequence[float]) -> Element:
    el = _sub(parent, tag)
    _sub(el, "xmin", str(extent[0]))
    _sub(el, "ymin", str(extent[1]))
    _sub(el, "xmax", str(extent[2]))
    _sub(el, "ymax", str(extent[3]))
    return el


def _fill_spatialrefsys(sys: Element, authid: str) -> Element:
    """Populate *sys* (a ``spatialrefsys`` element) for a known EPSG authid."""
    key = authid.upper()
    _sub(sys, "wkt", "")
    if key == "EPSG:27700":
        _sub(
            sys,
            "proj4",
            "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 "
            "+y_0=-100000 +ellps=airy +units=m +no_defs",
        )
        _sub(sys, "srsid", "2437")
        _sub(sys, "srid", "27700")
        _sub(sys, "authid", "EPSG:27700")
        _sub(sys, "description", "OSGB36 / British National Grid")
        _sub(sys, "projectionacronym", "tmerc")
        _sub(sys, "ellipsoidacronym", "EPSG:7001")
    elif key == "EPSG:3857":
        _sub(
            sys,
            "proj4",
            "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 "
            "+k=1 +units=m +nadgrids=@null +wktext +no_defs",
        )
        _sub(sys, "srsid", "3857")
        _sub(sys, "srid", "3857")
        _sub(sys, "authid", "EPSG:3857")
        _sub(sys, "description", "WGS 84 / Pseudo-Mercator")
        _sub(sys, "projectionacronym", "merc")
        _sub(sys, "ellipsoidacronym", "EPSG:7030")
    else:
        # Unknown EPSG: still emit authid/srid so QGIS can resolve on open.
        code = key.split(":", 1)[1]
        _sub(sys, "proj4", "")
        _sub(sys, "srsid", code)
        _sub(sys, "srid", code)
        _sub(sys, "authid", key)
        _sub(sys, "description", key)
        _sub(sys, "projectionacronym", "")
        _sub(sys, "ellipsoidacronym", "")
    _sub(sys, "geographicflag", "false")
    return sys


def _srs_for_authid(parent: Element, authid: str, tag: str = "srs") -> Element:
    """Attach ``<srs><spatialrefsys>…`` for a map layer."""
    srs = _sub(parent, tag)
    sys = _sub(srs, "spatialrefsys")
    _fill_spatialrefsys(sys, authid)
    return srs


def _append_suitability_renderer(pipe: Element) -> None:
    renderer = _sub(
        pipe,
        "rasterrenderer",
        band="1",
        classificationMin="0",
        classificationMax="1",
        alphaBand="-1",
        nodataColor="",
        opacity="0.9",
        type="singlebandpseudocolor",
    )
    _sub(renderer, "rasterTransparency")
    origin = _sub(renderer, "minMaxOrigin")
    _sub(origin, "limits", "None")
    _sub(origin, "extent", "WholeRaster")
    _sub(origin, "statAccuracy", "Estimated")
    _sub(origin, "cumulativeCutLower", "0.02")
    _sub(origin, "cumulativeCutUpper", "0.98")
    _sub(origin, "stdDevFactor", "2")
    shader = _sub(renderer, "rastershader")
    ramp = _sub(
        shader,
        "colorrampshader",
        labelPrecision="2",
        minimumValue="0",
        classificationMode="1",
        colorRampType="INTERPOLATED",
        maximumValue="1",
        clip="0",
    )
    for value, color, label in SUITABILITY_RAMP_ITEMS:
        _sub(
            ramp,
            "item",
            label=label,
            alpha="255",
            value=str(value),
            color=color,
        )


def _maplayer_raster(
    *,
    layer_id: str,
    name: str,
    datasource: str,
    provider: str = "gdal",
    with_suitability_ramp: bool = True,
    xyz: bool = False,
    raster_crs: str = DEFAULT_RASTER_CRS,
) -> Element:
    layer = Element(
        "maplayer",
        {
            "autoRefreshEnabled": "0",
            "autoRefreshTime": "0",
            "hasScaleBasedVisibilityFlag": "0",
            "legendPlaceholderImage": "",
            "maxScale": "0",
            "minScale": "1e+08",
            "refreshOnNotifyEnabled": "0",
            "refreshOnNotifyMessage": "",
            "styleCategories": "AllStyleCategories",
            "type": "raster",
        },
    )
    _extent_el(layer, "extent", DEFAULT_EXTENT)
    _sub(layer, "id", layer_id)
    _sub(layer, "datasource", datasource)
    _sub(layer, "keywordList")
    _sub(layer, "layername", name)
    # XYZ tiles are Web Mercator; prediction/cutoff rasters use the manifest CRS.
    _srs_for_authid(layer, BASEMAP_CRS if xyz else raster_crs)
    _sub(layer, "provider", provider)
    flags = _sub(layer, "flags")
    _sub(flags, "Identifiable", "1")
    _sub(flags, "Removable", "1")
    _sub(flags, "Searchable", "1")
    _sub(flags, "Private", "0")
    style_mgr = _sub(layer, "map-layer-style-manager", current="default")
    _sub(style_mgr, "map-layer-style", name="default")
    pipe = _sub(layer, "pipe")
    _sub(pipe, "provider", provider)
    if with_suitability_ramp and not xyz:
        _append_suitability_renderer(pipe)
    else:
        _sub(
            pipe,
            "rasterrenderer",
            opacity="1",
            alphaBand="-1",
            blueBand="-1",
            greenBand="-1",
            type="singlebandgray" if not xyz else "singlebandcolordata",
            grayBand="1",
            band="1",
        )
    _sub(layer, "blendMode", "0")
    _sub(layer, "opacity", "1")
    return layer


def _iter_species_groups(
    models: Sequence[ModelLayerSpec],
) -> list[tuple[str, list[ModelLayerSpec]]]:
    """Group models by species, preserving manifest order."""
    groups: list[tuple[str, list[ModelLayerSpec]]] = []
    index: dict[str, list[ModelLayerSpec]] = {}
    for model in models:
        group = species_group_name(model.model_id)
        if group not in index:
            index[group] = []
            groups.append((group, index[group]))
        index[group].append(model)
    return groups


def build_layers(manifest: ReviewManifest) -> list[BuiltLayer]:
    layers: list[BuiltLayer] = []
    for version in manifest.versions:
        for model in version.models:
            layers.append(
                BuiltLayer(
                    layer_id=_stable_layer_id(
                        version.version_id, model.model_id, "suitability"
                    ),
                    name=suitability_layer_name(model.model_id, version.version_id),
                    datasource=vsigs_path(model.run_folder, model.prediction_file),
                    provider="gdal",
                    kind="suitability",
                    version_id=version.version_id,
                    model_id=model.model_id,
                    species_group=species_group_name(model.model_id),
                    checked=False,
                )
            )
            if model.cutoff_file:
                layers.append(
                    BuiltLayer(
                        layer_id=_stable_layer_id(
                            version.version_id, model.model_id, "cutoff"
                        ),
                        name=cutoff_layer_name(model.model_id, version.version_id),
                        datasource=vsigs_path(model.run_folder, model.cutoff_file),
                        provider="gdal",
                        kind="cutoff",
                        version_id=version.version_id,
                        model_id=model.model_id,
                        species_group=species_group_name(model.model_id),
                        checked=False,
                    )
                )
    for name, datasource in BASEMAP_SPECS:
        layers.append(
            BuiltLayer(
                layer_id=_stable_layer_id("basemap", name),
                name=name,
                datasource=datasource,
                provider="wms",
                kind="basemap",
                version_id=None,
                model_id=None,
                species_group=None,
                checked=(name == "OpenStreetMap"),
            )
        )
    return layers


def _layer_tree(manifest: ReviewManifest, layers: Sequence[BuiltLayer]) -> Element:
    by_id = {layer.layer_id: layer for layer in layers}
    root = Element("layer-tree-group")
    group_paths: list[str] = []

    def add_layer_node(parent: Element, layer: BuiltLayer) -> None:
        checked = "Qt::Checked" if layer.checked else "Qt::Unchecked"
        _sub(
            parent,
            "layer-tree-layer",
            id=layer.layer_id,
            name=layer.name,
            source=layer.datasource,
            providerKey=layer.provider,
            checked=checked,
            expanded="0",
            legend_exp="",
            legend_split_behavior="0",
            patch_size="-1,-1",
        )

    for version in manifest.versions:
        vtitle = version_group_title(version)
        group_paths.append(vtitle)
        vgroup = _sub(
            root,
            "layer-tree-group",
            name=vtitle,
            checked="Qt::Checked" if version.version_id == "main" else "Qt::Unchecked",
            expanded="1" if version.version_id == "main" else "0",
            groupLayer="",
        )
        for species, models in _iter_species_groups(version.models):
            path = f"{vtitle}/{species}"
            group_paths.append(path)
            sgroup = _sub(
                vgroup,
                "layer-tree-group",
                name=species,
                checked="Qt::Checked",
                expanded="1",
                groupLayer="",
            )
            for model in models:
                suit_id = _stable_layer_id(
                    version.version_id, model.model_id, "suitability"
                )
                add_layer_node(sgroup, by_id[suit_id])
                if model.cutoff_file:
                    cut_id = _stable_layer_id(
                        version.version_id, model.model_id, "cutoff"
                    )
                    add_layer_node(sgroup, by_id[cut_id])

    basemap_group = _sub(
        root,
        "layer-tree-group",
        name="Basemaps",
        checked="Qt::Checked",
        expanded="0",
        groupLayer="",
    )
    group_paths.append("Basemaps")
    for layer in layers:
        if layer.kind == "basemap":
            add_layer_node(basemap_group, layer)

    # Attach collected paths for callers via attribute (set by build_project_xml).
    root.set("data-group-paths", "|".join(group_paths))
    return root


def _theme_visibility(
    *,
    name: str,
    layers: Sequence[BuiltLayer],
    visible_ids: set[str],
    checked_groups: Iterable[str],
) -> Element:
    preset = Element(
        "visibility-preset",
        {
            "name": name,
            "has-checked-group-info": "1",
            "has-expanded-info": "1",
        },
    )
    for layer in layers:
        visible = "1" if layer.layer_id in visible_ids else "0"
        _sub(
            preset,
            "layer",
            id=layer.layer_id,
            style="default",
            visible=visible,
            expanded="0",
        )
        _sub(preset, "expanded-legend-nodes", id=layer.layer_id)
    checked = _sub(preset, "checked-group-nodes")
    for group_id in checked_groups:
        _sub(checked, "checked-group-node", id=group_id)
    _sub(preset, "expanded-group-nodes")
    return preset


def _build_themes(
    manifest: ReviewManifest, layers: Sequence[BuiltLayer]
) -> tuple[list[Element], list[str]]:
    themes: list[Element] = []
    names: list[str] = []
    basemap_ids = {layer.layer_id for layer in layers if layer.kind == "basemap"}
    version_by_id = {version.version_id: version for version in manifest.versions}

    def version_layer_ids(version_id: str, *, suitability_only: bool = False) -> set[str]:
        out: set[str] = set()
        for layer in layers:
            if layer.version_id != version_id:
                continue
            if suitability_only and layer.kind != "suitability":
                continue
            out.add(layer.layer_id)
        return out

    main = version_by_id["main"]
    main_title = version_group_title(main)
    main_ids = version_layer_ids("main", suitability_only=True) | basemap_ids
    themes.append(
        _theme_visibility(
            name="Main",
            layers=layers,
            visible_ids=main_ids,
            checked_groups=[main_title, "Basemaps"],
        )
    )
    names.append("Main")

    compare = version_by_id.get("compare")
    if compare is not None:
        compare_title = version_group_title(compare)
        compare_ids = version_layer_ids("compare", suitability_only=True) | basemap_ids
        themes.append(
            _theme_visibility(
                name="Compare",
                layers=layers,
                visible_ids=compare_ids,
                checked_groups=[compare_title, "Basemaps"],
            )
        )
        names.append("Compare")

        # Per-model themes: one version at a time (Pair stacked both and hid one).
        compare_models = {model.model_id for model in compare.models}
        for model in main.models:
            if model.model_id not in compare_models:
                continue
            species = species_group_name(model.model_id)
            display = model_display_name(model.model_id)
            for version_id, vtitle in (
                ("main", main_title),
                ("compare", compare_title),
            ):
                theme_name = f"{display} — {version_short_tag(version_id)}"
                model_ids = {
                    layer.layer_id
                    for layer in layers
                    if layer.version_id == version_id
                    and layer.model_id == model.model_id
                } | basemap_ids
                themes.append(
                    _theme_visibility(
                        name=theme_name,
                        layers=layers,
                        visible_ids=model_ids,
                        checked_groups=[
                            vtitle,
                            f"{vtitle}/{species}",
                            "Basemaps",
                        ],
                    )
                )
                names.append(theme_name)

    return themes, names


def build_project_xml(manifest: ReviewManifest) -> BuiltProject:
    """Return deterministic plain-text ``.qgs`` XML for *manifest*."""
    layers = build_layers(manifest)
    root = Element(
        "qgis",
        {
            "projectname": manifest.title,
            "version": "3.34.0-Prizren",
        },
    )
    _sub(root, "homePath", path="")
    _sub(root, "title", manifest.title)
    _sub(root, "autoTransaction", "false")
    _sub(root, "evaluateDefaultValues", "false")
    _sub(root, "trustLayerMetadata", "false")

    tree = _layer_tree(manifest, layers)
    group_paths = tree.attrib.pop("data-group-paths", "").split("|")
    root.append(tree)

    project_layers = _sub(root, "projectlayers")
    for layer in layers:
        project_layers.append(
            _maplayer_raster(
                layer_id=layer.layer_id,
                name=layer.name,
                datasource=layer.datasource,
                provider=layer.provider,
                with_suitability_ramp=(layer.kind in {"suitability", "cutoff"}),
                xyz=(layer.kind == "basemap"),
                raster_crs=manifest.crs,
            )
        )

    layerorder = _sub(root, "layerorder")
    for layer in layers:
        _sub(layerorder, "layer", id=layer.layer_id)

    mapcanvas = _sub(root, "mapcanvas", name="theMapCanvas", annotationsVisible="1")
    _sub(mapcanvas, "units", "meters")
    _extent_el(mapcanvas, "extent", DEFAULT_EXTENT)
    _sub(mapcanvas, "rotation", "0")
    dest = _sub(mapcanvas, "destinationsrs")
    _fill_spatialrefsys(_sub(dest, "spatialrefsys"), manifest.crs)
    _sub(mapcanvas, "rendermaptile", "0")

    project_crs = _sub(root, "projectCrs")
    _fill_spatialrefsys(_sub(project_crs, "spatialrefsys"), manifest.crs)

    theme_els, theme_names = _build_themes(manifest, layers)
    presets = _sub(root, "visibility-presets")
    for theme_el in theme_els:
        presets.append(theme_el)

    properties = _sub(root, "properties")
    # QGIS 4.x needs this or project CRS stays empty despite <projectCrs>.
    spatial_ref = _sub(properties, "SpatialRefSys")
    _sub(spatial_ref, "ProjectionsEnabled", "1", type="int")
    themes_prop = _sub(properties, "Themes", type="Map")
    _sub(themes_prop, "Option", type="QString", name="defaultPreset", value="Main")

    xml_body = tostring(root, encoding="unicode")
    # Deterministic, diff-friendly plain text.
    xml = (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        + _pretty_xml(xml_body)
        + "\n"
    )
    return BuiltProject(
        xml=xml,
        layers=layers,
        theme_names=theme_names,
        group_paths=[p for p in group_paths if p],
    )


def _pretty_xml(xml: str) -> str:
    """Indent ElementTree output without depending on Python 3.9+ indent quirks."""
    # Normalize ElementTree's raw dump then indent by tag depth.
    compact = re.sub(r">\s+<", "><", xml.strip())
    parts = re.split(r"(?<=>)(?=<)", compact)
    lines: list[str] = []
    depth = 0
    for part in parts:
        if not part:
            continue
        if part.startswith("</"):
            depth = max(depth - 1, 0)
            lines.append(("  " * depth) + part)
        elif part.startswith("<?") or part.startswith("<!"):
            lines.append(part)
        elif part.endswith("/>") or re.match(r"<[^>]+>.*</[^>]+>$", part):
            lines.append(("  " * depth) + part)
        else:
            lines.append(("  " * depth) + part)
            if part.startswith("<") and not part.startswith("</"):
                depth += 1
    return "\n".join(lines)


def generate_project_text(manifest: ReviewManifest) -> str:
    return build_project_xml(manifest).xml


def write_project(
    manifest: ReviewManifest,
    *,
    repo_root: Path,
    output_path: Optional[Path] = None,
) -> Path:
    built = build_project_xml(manifest)
    out = output_path or (repo_root / manifest.project_relpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(built.xml, encoding="utf-8")
    return out


def find_credential_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in CREDENTIAL_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(match.group(0))
    for snippet in FORBIDDEN_BASEMAP_SNIPPETS:
        if snippet.lower() in text.lower():
            hits.append(snippet)
    return hits


def extract_layer_datasources(qgs_text: str) -> list[str]:
    """Return datasource values from maplayers and layer-tree sources."""
    found = re.findall(r"<datasource>([^<]*)</datasource>", qgs_text)
    found += re.findall(r'\bsource="([^"]*)"', qgs_text)
    # Deduplicate, preserve order. Unescape XML entities so allowlist
    # comparison matches the plain basemap URLs in BASEMAP_SPECS.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in found:
        decoded = html.unescape(item)
        if decoded not in seen:
            seen.add(decoded)
            ordered.append(decoded)
    return ordered


def assert_project_paths_allowed(
    qgs_text: str,
    manifest: ReviewManifest,
    *,
    extra_allowed: Optional[Iterable[str]] = None,
) -> None:
    """Fail when a layer path is outside the manifest + fixed basemaps.

    Published EV paths may be passed via *extra_allowed* when included later.
    """
    allowed = set(manifest_vsigs_paths(manifest))
    allowed |= fixed_basemap_datasources()
    if extra_allowed:
        allowed |= set(extra_allowed)
    for datasource in extract_layer_datasources(qgs_text):
        if datasource in allowed:
            continue
        if datasource.startswith("/vsigs/"):
            raise AssertionError(
                f"Unexpected /vsigs/ layer path not in manifest: {datasource}"
            )
        # XYZ / WMS basemap-style sources must be the fixed allowlist entries.
        if "type=xyz" in datasource or datasource.startswith("crs="):
            raise AssertionError(f"Unexpected basemap/WMS datasource: {datasource}")
        raise AssertionError(f"Unexpected layer datasource: {datasource}")


def assert_no_credentials(qgs_text: str) -> None:
    hits = find_credential_hits(qgs_text)
    if hits:
        raise AssertionError(f"Credential-like content in project: {sorted(set(hits))}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry: regenerate the committed review ``.qgs`` from the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / DEFAULT_MANIFEST_REL,
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
    out = write_project(manifest, repo_root=repo_root, output_path=args.output)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
