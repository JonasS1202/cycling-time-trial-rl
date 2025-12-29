import math
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import gpxpy
from fitparse import FitFile

# --------------------------------------------------------------------------- #
# GEOMETRY HELPERS
# --------------------------------------------------------------------------- #

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Distance between two (lat, lon) points in metres.
    """
    R_EARTH = 6371000.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates initial compass bearing (0-360 deg) between two points.
    0=North, 90=East, etc.
    """
    if lat1 == lat2 and lon1 == lon2: return 0.0
    
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    
    dlon = lon2_r - lon1_r
    
    y = math.sin(dlon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - \
        math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        
    theta_rad = math.atan2(y, x)
    bearing = (math.degrees(theta_rad) + 360) % 360
    return bearing

def smooth_bearings(bearings, window_size):
    """
    Smooths a sequence of bearings (degrees) using a rolling window 
    on the sine and cosine components.
    """
    if window_size < 2:
        return bearings
        
    rads = np.radians(bearings)
    sins = np.sin(rads)
    coss = np.cos(rads)
    
    # Create rolling window kernel
    kernel = np.ones(window_size) / window_size
    
    # Pad to handle edges (edge padding acts as holding the start/end bearing)
    pad_width = window_size // 2
    
    padded_sins = np.pad(sins, pad_width, mode='edge')
    padded_coss = np.pad(coss, pad_width, mode='edge')
    
    # Convolve
    smooth_sins = np.convolve(padded_sins, kernel, mode='valid')
    smooth_coss = np.convolve(padded_coss, kernel, mode='valid')
    
    # Trim to match original length exactly
    n = len(bearings)
    if len(smooth_sins) > n:
        smooth_sins = smooth_sins[:n]
        smooth_coss = smooth_coss[:n]
    
    # Reconstruct bearings
    smooth_rads = np.arctan2(smooth_sins, smooth_coss)
    smooth_bearings = (np.degrees(smooth_rads) + 360) % 360
    
    return smooth_bearings

# --------------------------------------------------------------------------- #
# FILE PARSING
# --------------------------------------------------------------------------- #

def _parse_fit(fit_path: str | Path) -> pd.DataFrame:
    """Extract track data from a .fit file."""
    fit = FitFile(str(fit_path))
    rows = []
    for rec in fit.get_messages("record"):
        ts   = rec.get_value("timestamp")
        dist = rec.get_value("distance")
        lat  = rec.get_value("position_lat")
        lon  = rec.get_value("position_long")
        alt  = rec.get_value("enhanced_altitude")
        pwr  = rec.get_value("power")
        spd  = rec.get_value("speed") # m/s
        
        if ts is not None:
            # FIT usually stores UTC. ensure awareness
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            
            rows.append((
                ts,
                dist if dist is not None else np.nan,
                lat * (180.0 / 2 ** 31) if lat is not None else np.nan,
                lon * (180.0 / 2 ** 31) if lon is not None else np.nan,
                alt if alt is not None else np.nan,
                pwr if pwr is not None else 0,
                spd if spd is not None else np.nan,
            ))
            
    df = pd.DataFrame(
        rows,
        columns=["time", "dist", "lat", "lon", "ele", "power", "speed"], # Renamed alt -> ele for consistency
    )
    
    # Drop rows with missing distance or altitude
    df = df.dropna(subset=['dist', 'ele'])
    
    # Ensure distance is strictly increasing
    df = df.sort_values('dist')
    df = df.drop_duplicates(subset=['dist'], keep='first')
    
    return df

def _parse_gpx(gpx_path: str | Path) -> pd.DataFrame:
    """Extract track data from a .gpx file."""
    with open(gpx_path, 'r') as f:
        gpx = gpxpy.parse(f)

    rows = []
    cumulative_dist = 0.0
    last_point = None
    
    all_points = [p for track in gpx.tracks for segment in track.segments for p in segment.points]
    
    if not all_points:
         raise ValueError(f"GPX file '{gpx_path}' contains no track points.")

    has_time = all_points[0].time is not None
    # If no time, assume start NOW (UTC)
    synthetic_time = dt.datetime.now(dt.timezone.utc)

    for i, point in enumerate(all_points):
        # We need lat, lon, ele
        if point.latitude is None or point.longitude is None:
            continue
            
        # Elevation might be None in some GPX, handle gracefully? 
        # For now assume required as per original code
        ele = point.elevation if point.elevation is not None else 0.0

        if last_point:
            cumulative_dist += haversine_distance(
                last_point.latitude, last_point.longitude,
                point.latitude, point.longitude
            )
        
        if has_time and point.time:
            timestamp = point.time
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
        else:
            timestamp = synthetic_time + dt.timedelta(seconds=i)

        rows.append((
            timestamp,
            cumulative_dist,
            point.latitude,
            point.longitude,
            ele,
            0, # power
            np.nan, # speed
        ))
        last_point = point

    if not rows:
        raise ValueError(f"GPX file '{gpx_path}' contains no valid track points.")

    df = pd.DataFrame(
        rows,
        columns=["time", "dist", "lat", "lon", "ele", "power", "speed"],
    )
    
    df = df.sort_values('dist')
    df = df.drop_duplicates(subset=['dist'], keep='first')
    return df

def load_track(track_path: str | Path) -> pd.DataFrame:
    """
    Universal track loader.
    Returns DataFrame: time, dist, lat, lon, ele, power, speed, bearing
    """
    path = Path(track_path)
    if not path.exists():
         raise FileNotFoundError(f"File {path} not found.")

    if path.suffix.lower() == ".fit":
        df = _parse_fit(path)
    elif path.suffix.lower() == ".gpx":
        df = _parse_gpx(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
        
    # Standardize: Add 'bearing' column if possible
    # Calculate Bearings
    bearings = []
    # Vectorized might be faster but loop is safe and robust
    lats = df.lat.values
    lons = df.lon.values
    
    for i in range(len(df)-1):
         b = calculate_bearing(lats[i], lons[i], lats[i+1], lons[i+1])
         bearings.append(b)
         
    if bearings:
         bearings.append(bearings[-1]) # Pad last
    else:
         bearings.append(0.0)
         
    df['bearing'] = bearings
    
    return df
