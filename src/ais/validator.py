"""
Validation Logic for AIS Records and Spatial-Temporal Boundaries
----------------------------------------------------------------
Validates coordinates, timestamps, MMSI identifiers, and navigational limits.
"""

from datetime import datetime
from typing import Tuple, List, Optional
from src.ais.schema import AISRecord


class AISValidationError(ValueError):
    """Raised when an AIS record contains physically or logically invalid data."""
    pass


def validate_ais_record(record: AISRecord) -> Tuple[bool, List[str]]:
    """
    Validates a single AISRecord against maritime physical constraints.
    
    Returns:
        (is_valid: bool, errors: List[str])
    """
    errors = []
    
    # 1. Coordinate boundaries
    if not (-90.0 <= record.latitude <= 90.0):
        errors.append(f"Latitude out of valid range [-90, 90]: got {record.latitude}")
        
    if not (-180.0 <= record.longitude <= 180.0):
        errors.append(f"Longitude out of valid range [-180, 180]: got {record.longitude}")
        
    # 2. MMSI check (standard MID format: 9 digits, typically 200000000 - 799999999)
    if not (100000000 <= record.mmsi <= 999999999):
        errors.append(f"MMSI must be a valid 9-digit maritime identifier: got {record.mmsi}")
        
    # 3. Speed Over Ground (SOG in knots; negative is invalid, > 102.2 indicates unavailable/error in AIS spec)
    if record.sog < 0.0 or record.sog > 102.2:
        errors.append(f"SOG out of plausible range [0.0, 102.2 knots]: got {record.sog}")
        
    # 4. Course Over Ground (COG in degrees [0.0, 360.0]; 360.0 means not available in standard AIS)
    if not (0.0 <= record.cog <= 360.0):
        errors.append(f"COG out of valid range [0.0, 360.0 degrees]: got {record.cog}")
        
    # 5. Heading (if present, must be [0, 359] or 511 indicating not available)
    if record.heading is not None:
        if not (0.0 <= record.heading <= 360.0 or record.heading == 511.0):
            errors.append(f"Heading out of valid range [0, 359] or 511: got {record.heading}")
            
    # 6. Nav status (if present, 0 to 15 in standard ITU-R M.1371)
    if record.nav_status is not None:
        if not (0 <= record.nav_status <= 15):
            errors.append(f"Nav status code out of range [0, 15]: got {record.nav_status}")
            
    # 7. Timestamp sanity check
    if not isinstance(record.timestamp, datetime):
        errors.append(f"Timestamp must be a valid datetime instance: got {type(record.timestamp)}")

    return (len(errors) == 0, errors)
