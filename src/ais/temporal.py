"""
Temporal Window Filtering for AIS Broadcast Records
---------------------------------------------------
Applies lookback and lookforward windows relative to spill event / origin release times.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
from src.ais.schema import AISRecord, SpillEvent


def get_spill_temporal_bounds(
    spill: SpillEvent,
    lookback_hours: float = 24.0,
    lookforward_hours: float = 6.0
) -> Tuple[datetime, datetime]:
    """
    Computes the start and end datetime bounds for AIS retrieval.
    
    If estimated origin release window is defined (origin_time_start / origin_time_end),
    anchors around the release window. Otherwise anchors around observation_time.
    """
    if spill.origin_time_start is not None:
        t_base_start = spill.origin_time_start
    else:
        t_base_start = spill.observation_time

    if spill.origin_time_end is not None:
        t_base_end = spill.origin_time_end
    elif spill.observation_time_end is not None:
        t_base_end = spill.observation_time_end
    else:
        t_base_end = spill.observation_time

    t_start = t_base_start - timedelta(hours=lookback_hours)
    t_end = t_base_end + timedelta(hours=lookforward_hours)
    return t_start, t_end


def filter_ais_by_time(
    records: List[AISRecord],
    t_start: datetime,
    t_end: datetime
) -> Tuple[List[AISRecord], Dict[str, Any]]:
    """
    Filters AIS records strictly within [t_start, t_end].
    
    Returns:
        (filtered_records, audit_stats)
    """
    passed = []
    dropped_before = 0
    dropped_after = 0

    for rec in records:
        if rec.timestamp < t_start:
            dropped_before += 1
        elif rec.timestamp > t_end:
            dropped_after += 1
        else:
            passed.append(rec)

    stats = {
        "total_input_records": len(records),
        "passed_records": len(passed),
        "dropped_before_start": dropped_before,
        "dropped_after_end": dropped_after,
        "window_start": t_start.isoformat(),
        "window_end": t_end.isoformat()
    }
    return passed, stats
