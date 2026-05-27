"""Fit an empirical wave-height model from SQLite history.

Physics motivation: in fetch-limited deep water, significant wave height
scales with wind speed squared (SMB / Pierson-Moskowitz). The fully-
developed-sea coefficient is ~0.0057 m/kt² (H_s = 0.21·U²/g with U in
m/s, converted to kt). In Howe Sound / Strait of Georgia the effective
coefficient is smaller because of limited fetch and the lag between
wind onset and wave build-up.

We resample readings to 1-hour buckets, then for each candidate lag L
in 0..12h fit:
    wave_m(t)  ≈  a · wind_kts(t − L)²  +  b
Pick the lag with the highest R². Report slope, intercept, R², lag,
sample count, and a one-line formula string.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from . import db

# 16-point → degrees, then snapped to nearest 45° (8-point band) so each
# bin has enough samples to fit.
_COMPASS_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}
_BAND8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _to_band(label):
    if not isinstance(label, str):
        return None
    label = label.strip().upper()
    deg = _COMPASS_DEG.get(label)
    if deg is None:
        return None
    idx = int(round(deg / 45.0)) % 8
    return _BAND8[idx]


def _load_df(buoy_id: str) -> pd.DataFrame:
    db.init()
    c = sqlite3.connect(str(db.DB_PATH), timeout=10)
    try:
        rows = c.execute("""
            SELECT ts, wind_speed, direction, wave_height_m FROM readings
            WHERE buoy_id=?
              AND wind_speed IS NOT NULL
              AND wave_height_m IS NOT NULL
              AND wave_height_m > 0
            ORDER BY ts
        """, (buoy_id,)).fetchall()
    finally:
        c.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "wind", "direction", "wave"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["ts"])
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce")
    df["band"] = df["direction"].apply(_to_band)
    return df.dropna(subset=["ts", "wind", "wave"])


def _fit_one(x_kts2, y_m):
    """Linear fit wave_m = slope * wind² + intercept. Returns (slope, intercept, r2) or None."""
    if len(x_kts2) < 5 or float(np.std(x_kts2)) == 0:
        return None
    slope, intercept = np.polyfit(x_kts2, y_m, 1)
    pred = slope * x_kts2 + intercept
    ss_res = float(np.sum((y_m - pred) ** 2))
    ss_tot = float(np.sum((y_m - y_m.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def fit(buoy_id: str, max_lag_hours: int = 12) -> dict:
    df_raw = _load_df(buoy_id)
    if len(df_raw) < 20:
        return {"error": f"need ≥20 paired readings, have {len(df_raw)}", "n_raw": len(df_raw)}

    # Hourly resample preserving the modal direction band per hour
    df_h = df_raw.set_index("ts")[["wind", "wave"]].resample("1h").mean()
    band_h = df_raw.set_index("ts")["band"].resample("1h").agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None
    )
    df = df_h.join(band_h.rename("band")).dropna(subset=["wind", "wave"])
    n_hourly = len(df)
    if n_hourly < 20:
        return {"error": f"after hourly resample only {n_hourly} rows",
                "n_raw": int(len(df_raw))}

    # ── Global fit: scan all lags, pick best by R² ─────────────────────
    results = []
    for lag in range(0, max_lag_hours + 1):
        pair = pd.DataFrame({
            "w": df["wind"].shift(lag),
            "h": df["wave"],
        }).dropna()
        if len(pair) < 10:
            continue
        f = _fit_one(pair["w"].values ** 2, pair["h"].values)
        if f is None:
            continue
        slope, intercept, r2 = f
        results.append({
            "lag_hours": lag,
            "n_pairs": int(len(pair)),
            "slope_m_per_kt2": round(slope, 6),
            "intercept_m": round(intercept, 4),
            "r2": round(r2, 4),
        })

    if not results:
        return {"error": "no valid lag fits"}

    best = max(results, key=lambda r: r["r2"])

    # ── Per-direction fit at the global best lag ──────────────────────
    # Same lag for every band so a forecast formula stays comparable.
    band_lag = best["lag_hours"]
    by_band = {}
    pair_global = pd.DataFrame({
        "w": df["wind"].shift(band_lag),
        "h": df["wave"],
        "band": df["band"].shift(band_lag),   # direction OF the past wind
    }).dropna()
    weighted_r2_num = 0.0
    weighted_r2_den = 0
    for band in _BAND8:
        sub = pair_global[pair_global["band"] == band]
        if len(sub) < 8:
            by_band[band] = {"n_pairs": int(len(sub)), "fitted": False}
            continue
        f = _fit_one(sub["w"].values ** 2, sub["h"].values)
        if f is None:
            by_band[band] = {"n_pairs": int(len(sub)), "fitted": False}
            continue
        slope, intercept, r2 = f
        by_band[band] = {
            "n_pairs": int(len(sub)),
            "slope_m_per_kt2": round(slope, 6),
            "intercept_m": round(intercept, 4),
            "r2": round(r2, 4),
            "fitted": True,
        }
        weighted_r2_num += r2 * len(sub)
        weighted_r2_den += len(sub)
    weighted_r2 = round(weighted_r2_num / weighted_r2_den, 4) if weighted_r2_den else None

    # SMB reference: 0.21 * (kt*0.5144)^2 / 9.81 ≈ 0.00566 m/kt²
    smb_slope = 0.21 * (0.5144 ** 2) / 9.81
    fetch_limit_pct = round(100.0 * best["slope_m_per_kt2"] / smb_slope, 1) if smb_slope else None

    return {
        "buoy_id": buoy_id,
        "n_hourly_rows": int(n_hourly),
        "best": best,
        "all_lags": results,
        "smb_reference_slope": round(smb_slope, 5),
        "fetch_limit_pct_of_open_ocean": fetch_limit_pct,
        "by_direction": by_band,
        "weighted_r2_across_directions": weighted_r2,
        "formula": (f"wave_m ≈ {best['slope_m_per_kt2']:.5f} × wind_kts² "
                    f"{'+' if best['intercept_m'] >= 0 else '−'} "
                    f"{abs(best['intercept_m']):.3f}   "
                    f"(use wind from {best['lag_hours']}h ago, R²={best['r2']:.2f})"),
    }


def predict(buoy_id: str, wind_kts: float, direction: Optional[str] = None) -> Optional[dict]:
    """Apply the best-fit model. If `direction` is given and that band has
    a fitted sub-model, use the per-direction params (much better signal
    in fetch-asymmetric basins); otherwise fall back to the global fit."""
    m = fit(buoy_id)
    if "error" in m:
        return None
    used = "global"
    slope = m["best"]["slope_m_per_kt2"]
    intercept = m["best"]["intercept_m"]
    r2 = m["best"]["r2"]
    band = _to_band(direction) if direction else None
    if band and m["by_direction"].get(band, {}).get("fitted"):
        bb = m["by_direction"][band]
        slope = bb["slope_m_per_kt2"]
        intercept = bb["intercept_m"]
        r2 = bb["r2"]
        used = f"direction:{band}"
    wave_m = slope * (wind_kts ** 2) + intercept
    return {
        "input_wind_kts": wind_kts,
        "input_direction": direction,
        "model_used": used,
        "predicted_wave_m": round(max(0.0, wave_m), 3),
        "predicted_wave_cm": round(max(0.0, wave_m * 100), 1),
        "lag_hours": m["best"]["lag_hours"],
        "r2": r2,
    }
