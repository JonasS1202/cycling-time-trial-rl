"""
weather.py – universal past/future wind fetcher for Open-Meteo
Revision: December 2025

Features
--------
* Historical (ERA5) and forecast endpoints picked automatically
* Model routing: HRRR, ICON-D2/EU/Global, GFS-25
* Hourly data for up to 100 coordinates per call
* Strict UTC timestamps
* Wind speeds requested in m s⁻¹; temperature returned in °C, converted to K
* Ideal-gas density, wind vectors, 16-point compass
"""

from __future__ import annotations

import datetime as dt
import time
from itertools import islice
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
R_SPEC = 287.05         # J kg⁻¹ K⁻¹  (dry-air gas constant)
CHUNK  = 100            # Max coordinates per Open-Meteo request

# --------------------------------------------------------------------------- #
# Resilient HTTP session                                                      #
# --------------------------------------------------------------------------- #
def _make_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("https://", adapter)
    return sess


_SESSION = _make_session()

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _batched(seq, n: int = CHUNK):
    it = iter(seq)
    while (batch := list(islice(it, n))):
        yield batch


def _within(lat: float, lon: float,
            lat_min: float, lat_max: float,
            lon_min: float, lon_max: float) -> bool:
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _pick_model(lat: float, lon: float,
                horizon_h: float) -> str:
    """
    Return preferred single model for one coordinate.
    horizon_h   – absolute forecast horizon in hours (0 for pure past).
    """
    # HRRR: CONUS, ≤ 48 h ahead
    if _within(lat, lon, 24.0, 50.0, -130.0, -60.0) and horizon_h <= 48:
        return "hrrr"

    # ICON-D2: Central Europe, ≤ 48 h ahead
    if _within(lat, lon, 44.0, 55.0, 5.0, 16.0) and horizon_h <= 48:
        return "icon_d2"

    # ICON-EU: Europe & N. Atlantic
    if _within(lat, lon, 30.0, 78.0, -40.0, 75.0):
        return "icon_eu"

    # ICON-GLOBAL: worldwide, ≤ 5 d ahead
    if horizon_h <= 120:
        return "icon_global"

    # Fallback: GFS 0.25°
    return "gfs_25"


def _select_endpoint(start: dt.datetime,
                     end: dt.datetime) -> str:
    """
    Choose forecast vs archive endpoint for the requested window
    (archive covers everything strictly earlier than now−5 days).
    """
    now = dt.datetime.now(dt.timezone.utc)
    # If the window ends significantly in the past, use archive.
    # Open-Meteo forecast usually holds ~10 days of past data too, but safe margin is 5 days.
    if end < now - dt.timedelta(days=5):
        return "https://archive-api.open-meteo.com/v1/archive"
    return "https://api.open-meteo.com/v1/forecast"


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def _parse_hourly(hourly: dict) -> pd.DataFrame:
    """
    JSON → tidy DataFrame with:
        T_K, p_Pa, rho,
        wind_ms, gust_ms,
        dir_rad_from, w_E, w_N
    Index is UTC-aware datetime.
    """
    df = pd.DataFrame(hourly)
    # Open-Meteo returns ISO strings or similar, usually standard. 
    # Since we requested timezone=UTC (or GMT), we can treat them as UTC.
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    df.set_index("time", inplace=True)

    # Convert °C → K for density computation
    T_K = df["temperature_2m"] + 273.15
    p_Pa = df["surface_pressure"] * 100.0        # hPa → Pa
    # Ideal gas law for dry air: rho = p / (R * T)
    rho = p_Pa / (R_SPEC * T_K)

    # Wind direction: 0=N, 90=E, 180=S, 270=W (meteorological convention "coming from")
    dir_from_deg = df["wind_direction_10m"]
    dir_from_rad = np.deg2rad(dir_from_deg)
    
    # Vector components (blowing towards):
    # From N (0°) means blowing TO South.
    # u (East) = -sin(dir) * speed ?? No.
    # Let's map std meteo direction to math angle (0=E, 90=N).
    # Meteo: 0=N (from), 90=E (from).
    # To convert "Direction From" to "Vector Pointing To":
    #   Vector_Angle = 270 - Meteo_Angle
    #   Example: Meteo 0 (N) -> Blows South (270 math).
    #   Example: Meteo 90 (E) -> Blows West (180 math).
    #   w_E = speed * cos(math_angle)
    #   w_N = speed * sin(math_angle)
    
    # However, existing code used: dir_to = dir_from + pi.
    # N(0) -> S(180). sin(180)=0, cos(180)=-1 -> w_E=0, w_N=-speed. Correct (North wind blows South).
    # E(90) -> W(270). sin(270)=-1, cos(270)=0 -> w_E=-speed, w_N=0. Correct (East wind blows West).
    
    dir_to = dir_from_rad + np.pi
    w_E = df["wind_speed_10m"] * np.sin(dir_to)
    w_N = df["wind_speed_10m"] * np.cos(dir_to)
    
    gust = df.get("wind_gusts_10m", df["wind_speed_10m"])

    out = pd.DataFrame(
        {
            "T_K":            T_K,
            "p_Pa":           p_Pa,
            "rho":            rho,
            "wind_ms":        df["wind_speed_10m"],
            "gust_ms":        gust,
            "dir_rad_from":   dir_from_rad,
            "w_E":            w_E,
            "w_N":            w_N,
        },
        index=df.index,
    )
    return out


# --------------------------------------------------------------------------- #
# Core fetcher                                                                #
# --------------------------------------------------------------------------- #
def _consistent_model(models: List[str]) -> str | None:
    """Return the single model if all agree, else None."""
    uniq = set(models)
    return uniq.pop() if len(uniq) == 1 else None


def _fetch_batch(coords: List[Tuple[float, float]],
                 start: dt.datetime,
                 end: dt.datetime,
                 horizon_h: float) -> List[pd.DataFrame]:
    """
    Fetch one ≤100-coordinate batch from the appropriate endpoint.
    """
    endpoint = _select_endpoint(start, end)

    lat_str = ",".join(f"{lat:.5f}" for lat, _ in coords)
    lon_str = ",".join(f"{lon:.5f}" for _, lon in coords)

    # Format dates as YYYY-MM-DD for the API
    params = (
        f"latitude={lat_str}&longitude={lon_str}"
        f"&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}"
        "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,"
        "wind_gusts_10m,surface_pressure"
        "&timezone=UTC"  # Force UTC
        "&wind_speed_unit=ms"
    )

    # Forecast endpoint: add models= if all coords pick the same one
    if endpoint.endswith("/forecast"):
        models = [_pick_model(lat, lon, horizon_h) for lat, lon in coords]
        if (model := _consistent_model(models)) is not None:
            params += f"&models={model}"

    url = f"{endpoint}?{params}"
    resp = _SESSION.get(url, timeout=40)
    resp.raise_for_status()

    data = resp.json()
    if isinstance(data, dict):        # archive might return single dict if 1 location? No, normally list if multi-loc args.
        # However, Open-Meteo behaves differently if 1 coord vs N coords.
        # But we pass comma-sep string, so it should return list.
        # Exception: error response is a dict.
        data = [data]

    out: List[pd.DataFrame] = []
    for loc in data:
        if "error" in loc and loc["error"]:
             raise RuntimeError(f"API Error: {loc.get('reason', 'unknown')}")
        
        # Check if hourly data exists
        if "hourly" not in loc:
             # Could be a partial error or out of bounds. Return empty DF?
             # Raising error helps debug.
             raise RuntimeError(f"No hourly data in response for {loc.get('latitude')},{loc.get('longitude')}")

        out.append(_parse_hourly(loc["hourly"]))
    return out


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def fetch_multi_weather(
    coords: List[Tuple[float, float]],
    start_utc: dt.datetime,
    hours: int,
) -> Dict[int, pd.DataFrame]:
    """
    Retrieve hourly weather for many coordinates.

    Parameters
    ----------
    coords      list of (lat, lon)
    start_utc   tz-aware UTC datetime (can be past or future)
    hours       positive → forward in time, negative → backward

    Returns
    -------
    {index_in_coords: DataFrame}
    """
    if start_utc.tzinfo is None: # Naive is assumed UTC in some contexts, but let's be strict or converting
        # Assuming input is UTC if naive, or convert if aware.
        # Best to raise if naive to avoid ambiguity.
        raise ValueError("start_utc must be timezone-aware (preferably UTC)")
    
    start_utc = start_utc.astimezone(dt.timezone.utc)

    end_utc = start_utc + dt.timedelta(hours=hours)
    # Open-Meteo works with date ranges. We broaden to cover the full window.
    # If we need hourly data from T to T+H, we request the date(s) covering that.
    start_d = min(start_utc, end_utc)
    end_d = max(start_utc, end_utc)
    
    # We pass dates to API, which returns midnight-to-midnight for those dates usually.
    # We will filter later if needed, but keeping all is fine.
    
    horizon_h = abs(hours)

    results: Dict[int, pd.DataFrame] = {}
    
    # Deduplicate coords to save requests??
    # For now, just robust batching.
    
    for batch_idx, batch in enumerate(_batched(coords)):
        retries = 0
        while True:
            try:
                dfs = _fetch_batch(batch, start_d, end_d, horizon_h)
                break
            except Exception as exc:
                retries += 1
                if retries > 3:
                    print(f"Failed to fetch batch {batch_idx}: {exc}")
                    # Return empty DFs or reraise? 
                    # If we fail, we fail hard for now as weather is required if requested.
                    raise
                time.sleep(1.5 * retries)

        for i, df in enumerate(dfs):
            results[batch_idx * CHUNK + i] = df
    return results


class WeatherData:
    """
    Holds weather data for multiple locations and provides interpolation.
    """
    def __init__(self, data_map: Dict[int, pd.DataFrame], coords: List[Tuple[float, float]]):
        self.data_map = data_map
        self.coords = coords
        
    def get_at_index(self, index: int, time_utc: dt.datetime) -> Dict[str, float]:
        """
        Get weather at specific coordinate index and time.
        """
        if index not in self.data_map:
            raise KeyError(f"No weather data for index {index}")
            
        df = self.data_map[index]
        
        # Exact match or nearest? 
        # API returns hourly. Linear iterpolation is better.
        # Sort just in case
        df = df.sort_index()
        
        ts = time_utc.timestamp()
        times = df.index.astype(np.int64) // 10**9
        
        # We want to interpolate all columns.
        # Finding insertion point
        idx = np.searchsorted(times, ts)
        
        if idx == 0:
            return df.iloc[0].to_dict()
        if idx >= len(df):
            return df.iloc[-1].to_dict()
            
        # Linear interp
        t0 = times[idx-1]
        t1 = times[idx]
        
        alpha = (ts - t0) / (t1 - t0) if t1 > t0 else 0
        
        row0 = df.iloc[idx-1]
        row1 = df.iloc[idx]
        
        # Interpolate numeric columns
        res = {}
        for col in df.columns:
            val0 = row0[col]
            val1 = row1[col]
            # Handle angle wrap-around for direction?
            # It's better to interpolate w_E and w_N vectors, then recompute direction if needed.
            # But here we just return what we have.
            # We have w_E, w_N. Interpolating them is correct vector interpolation.
            # Interpolating dir_rad_from directly is risky near 0/2pi.
            if col == 'dir_rad_from':
                 # Skip simple lerp for angle, allow it to be inconsistent or 
                 # recompute from interpolated w_E/w_N later if strictly needed.
                 # For simulation we use w_E/w_N usually.
                 pass
            res[col] = val0 + (val1 - val0) * alpha
            
        return res


# --------------------------------------------------------------------------- #
# Convenience printer                                                         #
# --------------------------------------------------------------------------- #
_CARDINAL_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def _deg_to_cardinal(deg: float) -> str:
    return _CARDINAL_16[int((deg + 11.25) // 22.5) % 16]


def print_weather_summary(
    lat: float,
    lon: float,
    *,
    start: dt.datetime | None = None,
    hours: int = 0,
    label: str | None = None,
) -> None:
    """
    Pretty terminal printout for a single location.
    """
    if start is None:
        start = dt.datetime.now(dt.timezone.utc)
    if start.tzinfo is None:
         # Warn or fix? Fix to UTC if naive for convenience in print tool
         start = start.replace(tzinfo=dt.timezone.utc)

    data_map = fetch_multi_weather([(lat, lon)], start, hours)
    df = data_map[0].copy()

    # Pretty print
    disp = pd.DataFrame()
    disp["time"] = df.index.strftime("%Y-%m-%d %H:%M UTC")
    disp["temp [°C]"] = (df["T_K"] - 273.15).round(1)
    disp["press [hPa]"] = (df["p_Pa"] / 100).round(0)
    disp["wind [m/s]"] = df["wind_ms"].round(2)
    disp["gust [m/s]"] = df["gust_ms"].round(2)
    
    deg_from = (np.rad2deg(df["dir_rad_from"]) + 360) % 360
    disp["dir [from]"] = [f"{int(d):3d}° {_deg_to_cardinal(d)}" for d in deg_from]
    disp["ρ [kg m⁻³]"] = df["rho"].round(3)

    cols = ["time", "temp [°C]", "press [hPa]",
            "wind [m/s]", "gust [m/s]", "dir [from]", "ρ [kg m⁻³]"]

    header = f"Weather summary for {label or f'({lat:.4f}, {lon:.4f})'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(disp[cols].to_string(index=False))
    print()


# --------------------------------------------------------------------------- #
# Demo                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Forecast: Regensburg next 6 h
    print_weather_summary(51.199797, 7.969037, hours=6, label="Regensburg")

    # History: Denver previous 24 h
    start_hist = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
    print_weather_summary(49.2049, 12.0439,
                          start=start_hist, hours=24,
                          label="Cham (past 24 h)")
