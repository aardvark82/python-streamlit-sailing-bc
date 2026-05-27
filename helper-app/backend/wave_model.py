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


def _load_df(buoy_id: str) -> pd.DataFrame:
    db.init()
    c = sqlite3.connect(str(db.DB_PATH), timeout=10)
    try:
        rows = c.execute("""
            SELECT ts, wind_speed, wave_height_m FROM readings
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
    df = pd.DataFrame(rows, columns=["ts", "wind", "wave"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["ts"])
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce")
    return df.dropna()


def fit(buoy_id: str, max_lag_hours: int = 12) -> dict:
    df = _load_df(buoy_id)
    if len(df) < 20:
        return {"error": f"need ≥20 paired readings, have {len(df)}", "n_raw": len(df)}

    # Hourly resample so lag in "rows" == hours exactly
    df = df.set_index("ts").resample("1h").mean().dropna()
    n_hourly = len(df)
    if n_hourly < 20:
        return {"error": f"after hourly resample only {n_hourly} rows",
                "n_raw": int(len(df))}

    results = []
    for lag in range(0, max_lag_hours + 1):
        # Positive lag = past wind predicts current wave
        pair = pd.DataFrame({
            "w": df["wind"].shift(lag),
            "h": df["wave"],
        }).dropna()
        if len(pair) < 10:
            continue
        x = pair["w"].values ** 2     # wind squared (kt²)
        y = pair["h"].values          # wave height (m)
        if x.std() == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results.append({
            "lag_hours": lag,
            "n_pairs": int(len(pair)),
            "slope_m_per_kt2": round(float(slope), 6),
            "intercept_m": round(float(intercept), 4),
            "r2": round(r2, 4),
        })

    if not results:
        return {"error": "no valid lag fits"}

    best = max(results, key=lambda r: r["r2"])

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
        "formula": (f"wave_m ≈ {best['slope_m_per_kt2']:.5f} × wind_kts² "
                    f"{'+' if best['intercept_m'] >= 0 else '−'} "
                    f"{abs(best['intercept_m']):.3f}   "
                    f"(use wind from {best['lag_hours']}h ago, R²={best['r2']:.2f})"),
    }


def predict(buoy_id: str, wind_kts: float) -> Optional[dict]:
    """Convenience: apply the best-fit model to a forecast wind speed."""
    m = fit(buoy_id)
    if "error" in m:
        return None
    b = m["best"]
    wave_m = b["slope_m_per_kt2"] * (wind_kts ** 2) + b["intercept_m"]
    return {
        "input_wind_kts": wind_kts,
        "predicted_wave_m": round(max(0.0, wave_m), 3),
        "predicted_wave_cm": round(max(0.0, wave_m * 100), 1),
        "lag_hours": b["lag_hours"],
        "r2": b["r2"],
    }
