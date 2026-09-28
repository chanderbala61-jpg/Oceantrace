"""
OceanTrace Professional SVG Icon System
=======================================
Lucide-style lightweight, consistent stroke SVG icons.
Replaces informal emoji decorators with scientific, technical iconography.
"""

def icon_svg(name: str, size: int = 16, color: str = "currentColor", stroke_width: float = 2.0) -> str:
    """Returns an inline SVG string for the requested icon name."""
    paths = {
        "grid": (
            f'<rect width="7" height="7" x="3" y="3" rx="1"/>'
            f'<rect width="7" height="7" x="14" y="3" rx="1"/>'
            f'<rect width="7" height="7" x="14" y="14" rx="1"/>'
            f'<rect width="7" height="7" x="3" y="14" rx="1"/>'
        ),
        "scan": (
            f'<path d="M3 7V5a2 2 0 0 1 2-2h2"/>'
            f'<path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
            f'<path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
            f'<path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
            f'<circle cx="12" cy="12" r="3"/>'
        ),
        "layers": (
            f'<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
            f'<polyline points="2 17 12 22 22 17"/>'
            f'<polyline points="2 12 12 17 22 12"/>'
        ),
        "route": (
            f'<circle cx="6" cy="19" r="3"/>'
            f'<path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/>'
            f'<circle cx="18" cy="5" r="3"/>'
        ),
        "ship": (
            f'<path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>'
            f'<path d="M19.38 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.94 5.34 2.81 7.76"/>'
            f'<path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/>'
            f'<path d="M12 10V2"/>'
            f'<path d="M12 2 9 5"/>'
        ),
        "shield_check": (
            f'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>'
            f'<path d="m9 12 2 2 4-4"/>'
        ),
        "file_text": (
            f'<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
            f'<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
            f'<path d="M10 9H8"/>'
            f'<path d="M16 13H8"/>'
            f'<path d="M16 17H8"/>'
        ),
        "cpu": (
            f'<rect width="16" height="16" x="4" y="4" rx="2"/>'
            f'<rect width="6" height="6" x="9" y="9" rx="1"/>'
            f'<path d="M15 2v2"/>'
            f'<path d="M15 20v2"/>'
            f'<path d="M2 15h2"/>'
            f'<path d="M2 9h2"/>'
            f'<path d="M20 15h2"/>'
            f'<path d="M20 9h2"/>'
            f'<path d="M9 2v2"/>'
            f'<path d="M9 20v2"/>'
        ),
        "sun": (
            f'<circle cx="12" cy="12" r="4"/>'
            f'<path d="M12 2v2"/>'
            f'<path d="M12 20v2"/>'
            f'<path d="m4.93 4.93 1.41 1.41"/>'
            f'<path d="m17.66 17.66 1.41 1.41"/>'
            f'<path d="M2 12h2"/>'
            f'<path d="M20 12h2"/>'
            f'<path d="m6.34 17.66-1.41 1.41"/>'
            f'<path d="m19.07 4.93-1.41 1.41"/>'
        ),
        "moon": (
            f'<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>'
        ),
        "satellite": (
            f'<path d="M13 7 9 3 5 7l4 4"/>'
            f'<path d="m17 11 4 4-4 4-4-4"/>'
            f'<path d="m8 12 4 4 6-6-4-4Z"/>'
            f'<path d="m16 8 3-3"/>'
            f'<path d="M9 21a6 6 0 0 0-6-6"/>'
        ),
        "map_pin": (
            f'<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>'
            f'<circle cx="12" cy="10" r="3"/>'
        ),
        "download": (
            f'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            f'<polyline points="7 10 12 15 17 10"/>'
            f'<line x1="12" x2="12" y1="15" y2="3"/>'
        ),
        "alert_triangle": (
            f'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
            f'<line x1="12" x2="12" y1="9" y2="13"/>'
            f'<line x1="12" x2="12.01" y1="17" y2="17"/>'
        ),
        "check_circle": (
            f'<circle cx="12" cy="12" r="10"/>'
            f'<path d="m9 12 2 2 4-4"/>'
        ),
        "info": (
            f'<circle cx="12" cy="12" r="10"/>'
            f'<line x1="12" x2="12" y1="16" y2="12"/>'
            f'<line x1="12" x2="12.01" y1="8" y2="8"/>'
        ),
        "compass": (
            f'<circle cx="12" cy="12" r="10"/>'
            f'<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
        ),
        "refresh": (
            f'<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
            f'<path d="M21 3v5h-5"/>'
            f'<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
            f'<path d="M8 16H3v5"/>'
        ),
        "crosshair": (
            f'<circle cx="12" cy="12" r="10"/>'
            f'<line x1="22" x2="18" y1="12" y2="12"/>'
            f'<line x1="6" x2="2" y1="12" y2="12"/>'
            f'<line x1="12" x2="12" y1="6" y2="2"/>'
            f'<line x1="12" x2="12" y1="22" y2="18"/>'
        )
    }
    inner = paths.get(name, paths["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block; vertical-align:middle; margin-right:6px;">{inner}</svg>'
    )
