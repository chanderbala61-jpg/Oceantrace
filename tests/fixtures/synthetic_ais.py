"""
Synthetic AIS Test Dataset Fixture for OceanTrace Unit Testing
==============================================================
NOTE / DISCLAIMER:
This dataset is STRICTLY SYNTHETIC and generated exclusively for software unit testing.
It does NOT represent real vessels, real spills, or real-world attribution claims.
"""

import os
import csv
from datetime import datetime, timedelta
from typing import Optional


def create_synthetic_ais_fixture(
    csv_path: str,
    base_time: Optional[datetime] = None,
    base_lat: float = 28.50,
    base_lon: float = -88.50,
):
    """
    Generates a controlled CSV file containing 4 distinct vessel scenarios
    relative to a spill event centered at (base_lat, base_lon) and base_time.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    headers = ["timestamp", "mmsi", "latitude", "longitude", "sog", "cog", "heading", "nav_status", "ship_name", "ship_type"]

    if base_time is None:
        base_time = datetime(2026, 9, 4, 12, 0, 0)

    # 1. Vessel 111222333 (High compatibility transit)
    # Passes directly crossing (base_lat, base_lon) around base_time
    for i, minute in enumerate([-120, -60, 0, 60, 120]):
        t = base_time + timedelta(minutes=minute)
        frac = (minute + 120) / 240.0
        lat = (base_lat - 0.10) + frac * 0.20
        lon = (base_lon - 0.10) + frac * 0.20
        rows.append([
            t.isoformat() + "Z", 111222333, round(lat, 5), round(lon, 5), 12.5, 45.0, 45.0, 0, "TEST_TANKER_ALPHA", "Tanker"
        ])

    # 2. Vessel 444555666 (Temporal mismatch: 3 days prior)
    t_mismatch = base_time - timedelta(days=3)
    for i, minute in enumerate([-60, 0, 60]):
        t = t_mismatch + timedelta(minutes=minute)
        rows.append([
            t.isoformat() + "Z", 444555666, base_lat, base_lon, 14.0, 90.0, 90.0, 0, "TEST_CARGO_BETA", "Cargo"
        ])

    # 3. Vessel 777888999 (Spatial mismatch: ~100 km south)
    for i, minute in enumerate([-60, 0, 60]):
        t = base_time + timedelta(minutes=minute)
        rows.append([
            t.isoformat() + "Z", 777888999, base_lat - 1.0, base_lon, 10.0, 180.0, 180.0, 0, "TEST_FISHING_GAMMA", "Fishing"
        ])

    # 4. Vessel 999000111 (Loitering near edge ~18 km east)
    for i, minute in enumerate([-90, -30, 30, 90]):
        t = base_time + timedelta(minutes=minute)
        rows.append([
            t.isoformat() + "Z", 999000111, base_lat, base_lon + 0.18, 1.2, 12.0, 15.0, 1, "TEST_TUG_DELTA", "Tug"
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return csv_path
