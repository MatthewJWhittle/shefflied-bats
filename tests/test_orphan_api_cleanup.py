"""Smoke pins for #51 orphan API cleanup (#47 items 3, 4, 7)."""

import importlib
import inspect

import pytest


def test_annotate_points_still_exported():
    from sdm.data.processing import annotate_points
    from sdm.data.processing.core import annotate_points as core_annotate

    assert annotate_points is core_annotate
    assert callable(annotate_points)


@pytest.mark.parametrize(
    "module_name,attr",
    [
        ("sdm.data.processing.core", "merge_environmental_layers"),
        ("sdm.data.processing.core", "process_occurrence_data"),
        ("sdm.data.processing.core", "process_background_data"),
        ("sdm.data.processing.core", "extract_environmental_data"),
        ("sdm.data", "merge_environmental_layers"),
        ("sdm.raster.utils", "aggregate_raster"),
        ("sdm.raster.utils", "tile_bounding_box"),
        ("sdm.raster.utils", "points_to_grid_squares"),
        ("sdm.raster.utils", "align_to_grid"),
        ("sdm.raster.utils", "calculate_offset"),
    ],
)
def test_deleted_orphan_symbols_absent(module_name, attr):
    mod = importlib.import_module(module_name)
    assert not hasattr(mod, attr), f"{module_name}.{attr} should be removed"


def test_ceh_processing_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sdm.data.ceh_processing")


def test_landcover_canonical_ceh_helpers_remain():
    from sdm.data.landcover import (
        define_broad_habitat_categories,
        get_ceh_land_cover_codes_v2023,
    )

    codes = get_ceh_land_cover_codes_v2023()
    cats = define_broad_habitat_categories()
    assert "1" in codes
    assert "Grassland" in cats
    assert inspect.isfunction(get_ceh_land_cover_codes_v2023)
