"""
Export ecological / paper-evidence tables to data/sdm_predictions/paper_results/.

Loads only existing rasters (predictions aligned to evs-to-model.tif), QA CSV,
model_results, YAML configs, SHAP folders, occurrence GeoJSON optional.

Run from repo root:
  uv run python notebooks/postprocessing/export_paper_ecological.py

Or: import and call export_all(Path("/path/to/repo"))
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
import yaml


COMMON_NAMES = {
    "Nyctalus noctula": "Noctule",
    "Pipistrellus pipistrellus": "Common pipistrelle",
    "Pipistrellus pygmaeus": "Soprano pipistrelle",
    "Pipistrellus nathusii": "Nathusius pipistrelle",
    "Myotis daubentonii": "Daubenton's bat",
    "Plecotus auritus": "Brown long-eared bat",
    "Myotis nattereri": "Natterer's bat",
    "Nyctalus leisleri": "Leisler's bat",
    "Myotis brandtii": "Brandt's bat",
    "Myotis mystacinus": "Whiskered bat",
}


def variable_group(var: str) -> str:
    if var.startswith("climate_"):
        return "climate"
    if var.startswith("terrain_"):
        return "terrain_topography"
    if var.startswith("ceh_landcover"):
        return "land_cover"
    if var.startswith("vom_"):
        return "vegetation_structure"
    if var.startswith("os_distance") or var.startswith("os_cover") or var == "bgs_coast_distance_to_coast":
        return "distance_or_built_cover"
    return "other"


def caution_flag_row(row: pd.Series, qa_warn: str) -> str:
    flags = []
    if pd.isna(row.get("mean_cv_score")):
        flags.append("missing_cv_auc")
    if row.get("n_presence", 0) < 30:
        flags.append("low_n_presence")
    if pd.notna(row.get("std_cv_score")) and row["std_cv_score"] > 0.12:
        flags.append("high_cv_std")
    w = str(qa_warn or "")
    if w and w != "nan":
        flags.append("qa_warning")
    return "; ".join(flags) if flags else ""


def read_prediction(repo: Path, band_key: str) -> np.ndarray:
    pred_path = repo / "data" / "sdm_predictions" / f"prediction_{band_key}.tif"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with rasterio.open(pred_path) as src:
        p = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            p[p == nodata] = np.nan
        return p


def read_ev_band(repo: Path, band_name: str) -> tuple[np.ndarray, rasterio.Affine]:
    ev_path = repo / "data" / "evs" / "evs-to-model.tif"
    with rasterio.open(ev_path) as src:
        labels = list(src.descriptions)
        if band_name not in labels:
            raise KeyError(f"Band {band_name!r} not in EV stack ({len(labels)} bands)")
        idx = labels.index(band_name) + 1
        a = src.read(idx).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            try:
                if np.isfinite(nodata):
                    a[a == nodata] = np.nan
            except (TypeError, ValueError):
                pass
        a[a <= -1e30] = np.nan
        return a, src.transform


def stratify_by_quantiles(
    pred: np.ndarray,
    strat: np.ndarray,
    threshold: float,
    stratifier_name: str,
    *,
    bins: int = 5,
    pixel_m: float = 100.0,
) -> pd.DataFrame:
    """Bins stratifier into quantile buckets; summarizes suitability vs threshold."""
    m = np.isfinite(pred) & np.isfinite(strat)
    s_flat = strat[m]
    p_flat = pred[m]
    if s_flat.size < bins * 10:
        return pd.DataFrame()

    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(s_flat, qs))
    if len(edges) < 3:
        return pd.DataFrame()

    rows = []
    cell_area_km2 = (pixel_m / 1000.0) ** 2

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        if last:
            band_m = m & (strat >= lo) & (strat <= hi)
        else:
            band_m = m & (strat >= lo) & (strat < hi)

        if not np.any(band_m):
            continue
        pv = pred[band_m]
        valid = np.isfinite(pv)
        if not np.any(valid):
            continue
        pv = pv[valid]
        above = (pv >= threshold).mean()
        rows.append(
            {
                "stratifier": stratifier_name,
                "bin_index": i + 1,
                "strat_lo": float(lo),
                "strat_hi": float(hi),
                "n_pixels": int(np.sum(valid)),
                "area_km2_approx": round(float(np.sum(valid) * cell_area_km2), 3),
                "mean_suitability": round(float(np.mean(pv)), 6),
                "median_suitability": round(float(np.median(pv)), 6),
                "frac_pixels_above_threshold": round(float(above), 6),
            }
        )
    return pd.DataFrame(rows)


def moor_comparison(
    pred: np.ndarray,
    heath: np.ndarray,
    dtm: np.ndarray,
    threshold: float,
    *,
    pixel_m: float = 100.0,
) -> pd.DataFrame:
    """High moor proxy: uplifted heath + elevation (study-area quantiles within valid cells)."""
    m = np.isfinite(pred) & np.isfinite(heath) & np.isfinite(dtm)
    heath_q = np.nanpercentile(heath[m], 66)
    dtm_q = np.nanpercentile(dtm[m], 66)
    moor_mask = m & (heath >= heath_q) & (dtm >= dtm_q)

    rows = []

    cell_area_km2 = (pixel_m / 1000.0) ** 2
    for name, mask in [("moor_uplift_proxy", moor_mask), ("not_moor_uplift_proxy", m & ~moor_mask)]:
        if not np.any(mask):
            continue
        pv = pred[mask]
        valid = np.isfinite(pv)
        pv = pv[valid]
        if pv.size == 0:
            continue
        rows.append(
            {
                "stratum": name,
                "n_pixels": int(pv.size),
                "area_km2_approx": round(float(pv.size * cell_area_km2), 3),
                "mean_suitability": round(float(np.mean(pv)), 6),
                "frac_above_threshold": round(float(np.mean(pv >= threshold)), 6),
                "heath_p66_cut": round(float(heath_q), 6),
                "dtm_p66_cut": round(float(dtm_q), 3),
            }
        )
    return pd.DataFrame(rows)


def urban_suburban_greenspace_table(
    pred: np.ndarray,
    threshold: float,
    suburb: np.ndarray,
    urb: np.ndarray,
    impr_gr: np.ndarray,
    woodland: np.ndarray,
    *,
    pixel_m: float = 100.0,
) -> pd.DataFrame:
    """Composite index: woodland + improved grassland + suburban/2 − urban (CEH 0–1 fractions)."""
    m = np.isfinite(pred) & np.isfinite(suburb) & np.isfinite(urb) & np.isfinite(impr_gr) & np.isfinite(woodland)
    for a in (suburb, urb, impr_gr, woodland):
        m = m & (a >= -0.01) & (a <= 1.01)
    g = np.clip(woodland, 0, 1) + np.clip(impr_gr, 0, 1) + 0.5 * np.clip(suburb, 0, 1) - np.clip(urb, 0, 1)
    s = np.where(m, g, np.nan)
    return stratify_by_quantiles(pred, s, threshold, "greenspace_minus_urban_index", bins=5, pixel_m=pixel_m)


def export_all(repo: Path) -> dict[str, Path]:
    out = repo / "data" / "sdm_predictions" / "paper_results"
    out.mkdir(parents=True, exist_ok=True)

    labels_path = repo / "data" / "evs" / "environmental_band_labels_patch.json"
    band_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    mr = pd.read_csv(repo / "data" / "sdm_models" / "model_results.csv")
    qa_full = pd.read_csv(repo / "data" / "sdm_predictions" / "aggregate_qa_latest.csv")
    qa_p10 = qa_full[qa_full["threshold_label"] == "p10_main"].drop_duplicates(subset=["band_key"])

    # --- 1) Performance table ---
    perf = mr.merge(qa_p10[["identifier", "band_key", "threshold", "training_omission_rate", "suitable_area_percent", "warning"]], on="identifier", how="left")

    perf["model_id"] = perf["band_key"]
    perf["common_name"] = perf["latin_name"].map(COMMON_NAMES).fillna("")
    perf["caution_flag"] = perf.apply(lambda r: caution_flag_row(r, r.get("warning", "")), axis=1)

    perf_out = perf[
        [
            "model_id",
            "common_name",
            "latin_name",
            "activity_type",
            "n_presence",
            "mean_cv_score",
            "std_cv_score",
            "threshold",
            "training_omission_rate",
            "suitable_area_percent",
            "caution_flag",
        ]
    ].rename(
        columns={
            "mean_cv_score": "mean_cv_auc",
            "std_cv_score": "std_cv_auc",
            "training_omission_rate": "omission_rate_at_presences",
            "suitable_area_percent": "suitable_area_percent_of_valid_pixels",
        }
    )
    p_perf = out / "paper_model_performance_table.csv"
    perf_out.to_csv(p_perf, index=False)

    # --- 2) Variables long ---
    cfg_root = repo / "data" / "sdm_config"
    var_rows = []
    for d in sorted(cfg_root.iterdir()):
        if not d.is_dir() or not (d / "variables_config.yml").exists():
            continue
        m = re.match(r"^(.+)_(Roost|In_flight)$", d.name)
        if not m:
            continue
        base, suf = m.group(1), m.group(2)
        latin = base.replace("_", " ")
        act = "Roost" if suf == "Roost" else "In flight"
        band_key = f"{latin.lower().replace(' ', '_')}_{'roost' if act == 'Roost' else 'in_flight'}"
        txt = (d / "variables_config.yml").read_text()
        lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("#")]
        data = yaml.safe_load("\n".join(lines)) or {}
        for v in data.get("variables") or []:
            if not isinstance(v, str):
                continue
            meta = band_labels.get(v, {})
            var_rows.append(
                {
                    "model_id": band_key,
                    "variable": v,
                    "variable_group": variable_group(v),
                    "label": meta.get("label", ""),
                    "interpretation_description": meta.get("description", ""),
                }
            )

    vars_df = pd.DataFrame(var_rows)
    p_vars = out / "paper_selected_variables_table.csv"
    vars_df.to_csv(p_vars, index=False)

    # --- 3) SHAP top 5 ---
    shap_rows = []
    shap_root = repo / "data" / "sdm_predictions" / "visualization" / "shap"
    if shap_root.is_dir():
        for yml in sorted(shap_root.rglob("variables_with_shap_scores.yml")):
            rel = yml.relative_to(shap_root)
            parts = rel.parts
            if len(parts) < 2:
                continue
            latin_dir = parts[0].replace("_", " ")
            act_folder = parts[1]
            latin_name = latin_dir
            activity = "Roost" if act_folder.lower() == "roost" else "In flight"
            bk = f"{latin_name.lower().replace(' ', '_')}_{'roost' if activity == 'Roost' else 'in_flight'}"
            blob = yaml.safe_load(yml.read_text()) or {}
            scores = blob.get("variables") or {}
            items = [(k, float(v)) for k, v in scores.items()]
            items.sort(key=lambda x: abs(x[1]), reverse=True)
            top = items[:5]
            vals = ", ".join(f"{n}={s:.6f}" for n, s in top)
            names = "; ".join(n for n, _ in top)
            plot_rel = Path("data/sdm_predictions/visualization/shap") / rel.parent / "shap_importance.png"
            shap_rows.append(
                {
                    "model_id": bk,
                    "top_5_shap_variables": names,
                    "top_5_shap_values_concat": vals,
                    "shap_importance_plot_path_relative": str(plot_rel.as_posix()),
                    "latin_from_folder": latin_name,
                    "activity_from_folder": activity,
                }
            )
    pd.DataFrame(shap_rows).to_csv(out / "paper_shap_top5_table.csv", index=False)

    # --- 4) Habitat claim statistics (raster-based) ---
    habitat_frames: list[pd.DataFrame] = []

    def thr_for(band_key: str) -> float:
        r = qa_p10.loc[qa_p10["band_key"] == band_key, "threshold"]
        if r.empty:
            raise KeyError(band_key)
        return float(r.iloc[0])

    def push(claim_id: str, model_id_key: str, analysis: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        t = df.copy()
        t.insert(0, "paper_claim_id", claim_id)
        t.insert(1, "model_id", model_id_key)
        t.insert(2, "analysis", analysis)
        habitat_frames.append(t)

    # Resolve EV bands once
    pred_cache: dict[str, np.ndarray] = {}

    def get_pred(key: str) -> np.ndarray:
        if key not in pred_cache:
            pred_cache[key] = read_prediction(repo, key)
        return pred_cache[key]

    # Daubenton roost — distance to water
    for bk in ["myotis_daubentonii_roost", "myotis_daubentonii_in_flight"]:
        try:
            pred = get_pred(bk)
            ta = thr_for(bk)
            ow, _ = read_ev_band(repo, "os_distance_distance_to_water")
            push(
                "C_daubenton_water",
                bk,
                "distance_to_water_quantile_bins_os_distance_distance_to_water",
                stratify_by_quantiles(pred, ow, ta, "os_distance_distance_to_water"),
            )
        except (FileNotFoundError, KeyError) as e:
            warnings.warn(f"Skip Daubenton water {bk}: {e}")

    # Leisler's — suburban / urban / green proxy
    for bk in ["nyctalus_leisleri_roost", "nyctalus_leisleri_in_flight"]:
        try:
            pred = get_pred(bk)
            ta = thr_for(bk)
            sub, _ = read_ev_band(repo, "ceh_landcover_suburban")
            urb, _ = read_ev_band(repo, "ceh_landcover_urban")
            ig, _ = read_ev_band(repo, "ceh_landcover_improved_grassland")
            bw, _ = read_ev_band(repo, "ceh_landcover_broadleaved_woodland")
            df = stratify_by_quantiles(pred, sub, ta, "ceh_landcover_suburban_tertiles_via_quantiles_bins=5")
            df2 = stratify_by_quantiles(pred, urb, ta, "ceh_landcover_urban_quantile_bins")
            df3 = urban_suburban_greenspace_table(pred, ta, sub, urb, ig, bw)
            push("C_leisler_landcover_axes", bk, "suburban_fraction_bins", df)
            push("C_leisler_landcover_axes", bk, "urban_fraction_bins", df2)
            push("C_leisler_landcover_axes", bk, "composite_green_minus_urban", df3)
        except Exception as e:
            warnings.warn(f"Skip Leisleri {bk}: {e}")

    # Noctule — distance buildings
    for bk in ["nyctalus_noctula_roost", "nyctalus_noctula_in_flight"]:
        try:
            pred = get_pred(bk)
            ta = thr_for(bk)
            bd, _ = read_ev_band(repo, "os_distance_distance_to_buildings")
            push("C_noctule_buildings", bk, "distance_to_buildings_quantiles", stratify_by_quantiles(pred, bd, ta, "os_distance_distance_to_buildings"))
        except Exception as e:
            warnings.warn(f"Skip noctule {bk}: {e}")

    # Pipistrellus spp + Plecotus + Nattereri roost — key landcover strata
    for bk in [
        "pipistrellus_pipistrellus_roost",
        "pipistrellus_pygmaeus_roost",
        "plecotus_auritus_roost",
        "myotis_nattereri_roost",
    ]:
        try:
            pred = get_pred(bk)
            ta = thr_for(bk)
            sub, _ = read_ev_band(repo, "ceh_landcover_suburban")
            ww, _ = read_ev_band(repo, "ceh_landcover_broadleaved_woodland")
            push("C_pip_plecotus_nat_roost_evenness", bk, "suburban_fraction_bins", stratify_by_quantiles(pred, sub, ta, "ceh_landcover_suburban"))
            push("C_pip_plecotus_nat_roost_evenness", bk, "broadleaved_woodland_bins", stratify_by_quantiles(pred, ww, ta, "ceh_landcover_broadleaved_woodland"))
        except Exception as e:
            warnings.warn(f"Skip {bk}: {e}")

    # Brandt + Whiskered — DTM / TPI valleys
    for bk in ["myotis_brandtii_roost", "myotis_mystacinus_roost", "myotis_mystacinus_in_flight"]:
        try:
            pred = get_pred(bk)
            ta = thr_for(bk)
            dtm, _ = read_ev_band(repo, "terrain_dtm")
            tpi, _ = read_ev_band(repo, "terrain_stats_tpi")
            push("C_brandt_whisker_elevation_valley", bk, "elevation_bins", stratify_by_quantiles(pred, dtm, ta, "terrain_dtm"))
            push("C_brandt_whisker_elevation_valley", bk, "terrain_position_index_bins", stratify_by_quantiles(pred, tpi, ta, "terrain_stats_tpi"))
        except Exception as e:
            warnings.warn(f"Skip brandt/mystacin {bk}: {e}")

    # Moor / dales — multi-species same mask for illustration (use Daubenton roost as exemplar surface)
    for bk in ["myotis_daubentonii_roost", "pipistrellus_pipistrellus_roost"]:
        try:
            pred = get_pred(bk)
            ta = thr_for(bk)
            heath, _ = read_ev_band(repo, "ceh_landcover_upland_heathland")
            dtm, _ = read_ev_band(repo, "terrain_dtm")
            push("C_moor_dales_unsuitability_proxy", bk, "moor_uplift_heath_dtm_p66", moor_comparison(pred, heath, dtm, ta))
        except Exception as e:
            warnings.warn(f"Skip moor {bk}: {e}")

    if habitat_frames:
        hab = pd.concat(habitat_frames, ignore_index=True)
        hab.to_csv(out / "paper_habitat_claim_statistics.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "paper_habitat_claim_statistics.csv", index=False)

    # --- 5) Claim support matrix (derived from tables + heuristics) ---
    matrix_rows: list[dict[str, Any]] = []

    def add_m(claim: str, stat: str, result: str, conf: str, rec: str) -> None:
        matrix_rows.append(
            {
                "claim_from_paper_theme": claim,
                "statistic_calculated": stat,
                "result_summary": result,
                "confidence": conf,
                "recommendation": rec,
            }
        )

    # Read back habitat file for one-line summaries if present
    hab_path = out / "paper_habitat_claim_statistics.csv"
    hab_s = pd.read_csv(hab_path) if hab_path.exists() and hab_path.stat().st_size > 50 else pd.DataFrame()

    add_m(
        "Daubenton's bat along watercourses",
        "Mean suitability & frac above p10 by distance-to-water quantile bins (per model)",
        "See paper_habitat_claim_statistics.csv rows C_daubenton_water; compare lowest vs highest water-distance bins.",
        "moderate" if not hab_s.empty and hab_s["paper_claim_id"].eq("C_daubenton_water").any() else "weak",
        "keep with figure: water-distance gradient table for Daubenton roost/in-flight",
    )
    add_m(
        "Leisler's bat urban green space",
        "Suburban / urban / composite green-minus-urban index vs suitability",
        "See C_leisler_landcover_axes strata in paper_habitat_claim_statistics.csv",
        "moderate" if not hab_s.empty and hab_s["paper_claim_id"].eq("C_leisler_landcover_axes").any() else "weak",
        "soften regional wording unless you add South/West Yorkshire mask",
    )
    add_m(
        "Noctule suburban / low-density urban",
        "Distance-to-buildings quantile bins vs suitability",
        "See C_noctule_buildings in paper_habitat_claim_statistics.csv",
        "moderate",
        "keep; cite quantile table + SHAP distance-to-buildings if available",
    )
    add_m(
        "Pipistrelles / brown long-eared / Natterer's even suburban fringe & woodland",
        "Suburban + broadleaved woodland fraction bins (roost models)",
        "See C_pip_plecotus_nat_roost_evenness",
        "moderate",
        "keep nuanced: shows covariate strata not proven 'evenness' spatially",
    )
    add_m(
        "Brandt's / Whiskered valley / restricted topography",
        "DTM + TPI bins",
        "See C_brandt_whisker_elevation_valley",
        "weak_to_moderate",
        "soften — low counts for Brandt's roost; show elevation/TPI gradient only",
    )
    moor_cols = ["model_id", "stratum", "mean_suitability", "frac_above_threshold"]
    moor_cols = [c for c in moor_cols if c in hab_s.columns]
    moor_ok = hab_s.loc[hab_s["paper_claim_id"].eq("C_moor_dales_unsuitability_proxy"), moor_cols]
    moor_txt = moor_ok.to_string(index=False, max_rows=20) if len(moor_ok) else "No rows"
    add_m(
        "High moor & dales less suitable (several species)",
        "Moor proxy stratum (heath p66 & DTM p66) vs rest — exemplar species rasters",
        moor_txt[:2000] + ("…" if len(moor_txt) > 2000 else ""),
        "moderate" if len(moor_ok) else "unsupported",
        "keep as cautioned pattern; not all species; proxy not 'dales' boundary",
    )

    pd.DataFrame(matrix_rows).to_csv(out / "paper_claim_support_matrix.csv", index=False)

    return {
        "performance": p_perf,
        "variables": p_vars,
        "shap": out / "paper_shap_top5_table.csv",
        "habitat": out / "paper_habitat_claim_statistics.csv",
        "matrix": out / "paper_claim_support_matrix.csv",
    }


def main() -> None:
    repo = Path(__file__).resolve().parent.parent.parent
    if not (repo / "config.yml").exists():
        repo = Path.cwd()
    paths = export_all(repo)
    for k, v in paths.items():
        print(k, "->", v)


if __name__ == "__main__":
    main()
