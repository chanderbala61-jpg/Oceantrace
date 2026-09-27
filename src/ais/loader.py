"""
AIS Data Loader: CSV Ingestion with Robust Schema Mapping and Parsing
---------------------------------------------------------------------
Loads raw AIS records from CSV files, handles variable header naming conventions,
and validates records using the schema validator.
"""

import os
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from src.ais.schema import AISRecord
from src.ais.validator import validate_ais_record


# Common column name variations across global AIS repositories (MarineCadastre, AISHub, etc.)
HEADER_ALIASES = {
    "timestamp": ["timestamp", "basedatetime", "time", "datetime", "date_time_utc", "t"],
    "mmsi": ["mmsi", "vessel_mmsi", "mmsi_number"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "long", "x"],
    "sog": ["sog", "speed", "speed_over_ground", "velocity"],
    "cog": ["cog", "course", "course_over_ground"],
    "heading": ["heading", "true_heading", "th"],
    "nav_status": ["nav_status", "status", "navigation_status", "navstatus"],
    "ship_name": ["ship_name", "vessel_name", "name", "vesselname"],
    "ship_type": ["ship_type", "vessel_type", "shiptype"],
    "imo": ["imo", "imo_number"],
    "length": ["length", "vessel_length"],
    "width": ["width", "vessel_width", "beam"],
    "draught": ["draught", "draft"]
}


def _parse_timestamp(val: str) -> datetime:
    """Parses various standard ISO and common datetime formats."""
    val = val.strip()
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            pass
    # Fallback to datetime.fromisoformat
    return datetime.fromisoformat(val.replace("Z", "+00:00").split("+")[0])


def _find_header_map(fieldnames: List[str]) -> Dict[str, str]:
    """Maps normalized schema fields to the exact CSV column header found."""
    lower_map = {fn.strip().lower(): fn for fn in fieldnames}
    mapping = {}
    for standard_field, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                mapping[standard_field] = lower_map[alias]
                break
    return mapping


def load_ais_csv(
    filepath: str,
    strict: bool = False
) -> Tuple[List[AISRecord], Dict[str, Any]]:
    """
    Loads AIS records from a CSV file.
    
    Args:
        filepath: Path to CSV file.
        strict: If True, raises on invalid rows. If False, skips invalid rows and logs count.
        
    Returns:
        (valid_records: List[AISRecord], summary: Dict[str, Any])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"AIS file not found: {filepath}")

    records: List[AISRecord] = []
    total_rows = 0
    invalid_rows = 0
    parse_errors = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], {"total_rows": 0, "valid_records": 0, "invalid_rows": 0}

        hmap = _find_header_map(reader.fieldnames)
        
        # Verify required headers exist
        required_fields = ["timestamp", "mmsi", "latitude", "longitude", "sog", "cog"]
        missing = [rf for rf in required_fields if rf not in hmap]
        if missing:
            raise ValueError(f"CSV missing mandatory AIS fields: {missing}. Available headers: {reader.fieldnames}")

        for row_idx, row in enumerate(reader):
            total_rows += 1
            try:
                # Extract mandatory fields
                t_str = row[hmap["timestamp"]]
                ts = _parse_timestamp(t_str)
                mmsi = int(float(row[hmap["mmsi"]].strip()))
                lat = float(row[hmap["latitude"]].strip())
                lon = float(row[hmap["longitude"]].strip())
                sog = float(row[hmap["sog"]].strip())
                cog = float(row[hmap["cog"]].strip())

                # Extract optional fields
                heading = None
                if "heading" in hmap and row.get(hmap["heading"]):
                    try:
                        heading = float(row[hmap["heading"]].strip())
                    except ValueError:
                        pass

                nav_status = None
                if "nav_status" in hmap and row.get(hmap["nav_status"]):
                    try:
                        nav_status = int(float(row[hmap["nav_status"]].strip()))
                    except ValueError:
                        pass

                ship_name = row.get(hmap["ship_name"]).strip() if "ship_name" in hmap and row.get(hmap["ship_name"]) else None
                ship_type = row.get(hmap["ship_type"]).strip() if "ship_type" in hmap and row.get(hmap["ship_type"]) else None

                rec = AISRecord(
                    timestamp=ts,
                    mmsi=mmsi,
                    latitude=lat,
                    longitude=lon,
                    sog=sog,
                    cog=cog,
                    heading=heading,
                    nav_status=nav_status,
                    ship_name=ship_name,
                    ship_type=ship_type
                )

                is_valid, errs = validate_ais_record(rec)
                if is_valid:
                    records.append(rec)
                else:
                    invalid_rows += 1
                    if len(parse_errors) < 10:
                        parse_errors.append(f"Row {row_idx}: {errs}")
                    if strict:
                        raise ValueError(f"Invalid AIS record at row {row_idx}: {errs}")

            except Exception as e:
                invalid_rows += 1
                if len(parse_errors) < 10:
                    parse_errors.append(f"Row {row_idx}: {e}")
                if strict:
                    raise

    summary = {
        "total_rows_read": total_rows,
        "valid_records_loaded": len(records),
        "invalid_rows_skipped": invalid_rows,
        "sample_errors": parse_errors
    }
    return records, summary
