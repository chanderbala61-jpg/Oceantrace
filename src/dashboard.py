"""
OceanTrace Investigation Dashboard and Evidence Visualization Module
=====================================================================
Generates multi-panel visual investigation charts and standalone HTML reports
integrating:
  1. Sentinel-1 SAR Dual-Channel Imagery + Oil Spill Mask & Boundary
  2. SAR Radiometric Look-Alike & Ambiguity Assessment
  3. Backward Environmental Drift Hindcast Trajectory & Probable Source Zone
  4. AIS Vessel Trajectories, Closest Approach Vectors, and Compatibility Evidence
  5. Scientific Integrity Disclaimers and Missing-Data Indicators
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

from src.pipeline import InvestigationRecord
from src.ais.trajectory import VesselTrajectory


def render_investigation_dashboard(
    record: InvestigationRecord,
    sar_image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    vessel_trajectories: Optional[Dict[int, VesselTrajectory]] = None,
    output_path: Optional[str] = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Renders a unified 4-panel OceanTrace Investigation Dashboard.

    Panels:
      [Top-Left]  : SAR Backscatter with Spill Boundary Polygon & Centroid
      [Top-Right] : SAR Look-Alike & Radiometric Ambiguity Diagnostics
      [Bottom-Left] : Geospatial Backward Drift Hindcast & Probable Source Zone
      [Bottom-Right]: AIS Vessel Traffic Tracks & Candidate Compatibility Evidence
    """
    fig = plt.figure(figsize=(18, 12), dpi=dpi)
    fig.patch.set_facecolor("#0b111e")

    # Grid layout: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22, left=0.06, right=0.96, top=0.91, bottom=0.07)
    ax_sar = fig.add_subplot(gs[0, 0])
    ax_lookalike = fig.add_subplot(gs[0, 1])
    ax_hindcast = fig.add_subplot(gs[1, 0])
    ax_ais = fig.add_subplot(gs[1, 1])

    title_text = f"OCEANTRACE INCIDENT INVESTIGATION: {record.scene_id}"
    fig.suptitle(title_text, fontsize=16, fontweight="bold", color="#f0f6fc", y=0.97)

    # -------------------------------------------------------------
    # PANEL 1: SAR Backscatter & Slick Boundary
    # -------------------------------------------------------------
    ax_sar.set_facecolor("#161b22")
    for spine in ax_sar.spines.values():
        spine.set_color("#30363d")

    if sar_image is not None and sar_image.size > 0:
        # Display grayscale SAR backscatter (prefer VV channel 1 if dual-pol, else channel 0)
        img_disp = sar_image[..., 1] if (sar_image.ndim == 3 and sar_image.shape[-1] >= 2) else (
            sar_image[..., 0] if sar_image.ndim == 3 else sar_image
        )
        vmin = np.percentile(img_disp, 2)
        vmax = np.percentile(img_disp, 98)
        ax_sar.imshow(img_disp, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

        if mask is not None and np.sum(mask > 0.5) > 0:
            # Translucent mask overlay (green)
            mask_rgba = np.zeros((*mask.shape[:2], 4), dtype=np.float32)
            mask_rgba[mask > 0.5] = [0.0, 1.0, 0.4, 0.45]
            ax_sar.imshow(mask_rgba, origin="upper")

        ax_sar.set_title("1. Sentinel-1 SAR Backscatter & Slick Boundary", color="#58a6ff", fontsize=12, fontweight="bold")
    else:
        ax_sar.text(
            0.5, 0.5,
            "SAR Imagery Not Provided\n(Detection Mask Utilized)",
            color="#8b949e", ha="center", va="center", fontsize=11, transform=ax_sar.transAxes
        )
        ax_sar.set_title("1. Sentinel-1 SAR Backscatter", color="#58a6ff", fontsize=12, fontweight="bold")

    if record.spill:
        s = record.spill
        info_str = f"Observed Centroid: {s.centroid_lat:.4f}°N, {s.centroid_lon:.4f}°E\nSurface Area: {s.area_km2:.2f} km²\nObserved: {s.observation_time.strftime('%Y-%m-%d %H:%M UTC')}"
        ax_sar.text(
            0.03, 0.05, info_str,
            color="#ffffff", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#0d1117", ec="#238636", alpha=0.85),
            transform=ax_sar.transAxes
        )
    ax_sar.tick_params(colors="#8b949e")

    # -------------------------------------------------------------
    # PANEL 2: SAR Look-Alike & Ambiguity Diagnostics
    # -------------------------------------------------------------
    ax_lookalike.set_facecolor("#161b22")
    for spine in ax_lookalike.spines.values():
        spine.set_color("#30363d")
    ax_lookalike.set_title("2. Look-Alike Ambiguity & Radiometric Diagnostics", color="#58a6ff", fontsize=12, fontweight="bold")

    la = record.look_alike_assessment
    if la is not None:
        # Build diagnostic bar chart of evidence parameters
        metrics = ["Contrast (dB)", "Edge Gradient", "Aspect Ratio", "Compactness"]
        raw_vals = [
            la.damping_contrast_db if la.damping_contrast_db is not None else 0.0,
            la.edge_gradient_mean * 100.0 if la.edge_gradient_mean is not None else 0.0,
            min(la.aspect_ratio if la.aspect_ratio is not None else 1.0, 15.0),
            la.compactness * 10.0 if la.compactness is not None else 0.0
        ]
        bars = ax_lookalike.barh(metrics, raw_vals, color=["#238636", "#1f6feb", "#d29922", "#8957e5"], alpha=0.85, height=0.5)
        ax_lookalike.set_xlim(0, max(max(raw_vals) * 1.25, 10.0))
        ax_lookalike.tick_params(colors="#c9d1d9")

        for bar in bars:
            w = bar.get_width()
            ax_lookalike.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", color="#f0f6fc", fontsize=9)

        # Classification badge
        badge_color = "#f85149" if la.is_look_alike_suspect else "#2ea043"
        status_label = "LOOK-ALIKE SUSPECT" if la.is_look_alike_suspect else "TRUE OIL SPILL CANDIDATE"
        ax_lookalike.text(
            0.5, 0.90, f"Assessment: {status_label}\nAmbiguity Score: {la.ambiguity_score:.2f} / 1.0",
            color="#ffffff", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=badge_color, ec="none", alpha=0.9),
            transform=ax_lookalike.transAxes
        )

        # Flags and notes
        flags_text = "Ambiguity Flags:\n" + ("\n".join(f"• {f}" for f in la.flags) if la.flags else "• None (Clean high-confidence profile)")
        ax_lookalike.text(
            0.05, 0.10, flags_text,
            color="#c9d1d9", fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0d1117", ec="#30363d", alpha=0.8),
            transform=ax_lookalike.transAxes
        )
    else:
        ax_lookalike.text(
            0.5, 0.5,
            "Look-Alike Analysis Skipped\n(SAR Radiometric Data Not Provided)",
            color="#8b949e", ha="center", va="center", fontsize=11, transform=ax_lookalike.transAxes
        )

    # -------------------------------------------------------------
    # PANEL 3: Geospatial Drift Hindcast & Source Zone
    # -------------------------------------------------------------
    ax_hindcast.set_facecolor("#161b22")
    for spine in ax_hindcast.spines.values():
        spine.set_color("#30363d")
    ax_hindcast.set_title("3. Backward Environmental Drift Hindcast & Source Zone", color="#58a6ff", fontsize=12, fontweight="bold")

    if record.hindcast_result and record.spill:
        h = record.hindcast_result
        spill = record.spill

        # Plot observed centroid
        ax_hindcast.scatter([spill.centroid_lon], [spill.centroid_lat], c="#3fb950", s=140, marker="o", label="Observed Slick Centroid", zorder=5)

        # Plot backward trajectory
        if len(h.trajectory) >= 2:
            lons = [p.longitude for p in h.trajectory]
            lats = [p.latitude for p in h.trajectory]
            ax_hindcast.plot(lons, lats, color="#58a6ff", linestyle="--", linewidth=2.0, label="Backward Drift Vector", zorder=3)
            # Intermediate hourly drift nodes
            ax_hindcast.scatter(lons[1:-1], lats[1:-1], c="#79c0ff", s=30, marker="x", zorder=4)

        # Plot estimated origin
        ax_hindcast.scatter([h.estimated_origin_lon], [h.estimated_origin_lat], c="#f85149", s=180, marker="*", label="Estimated Release Origin", zorder=6)

        # Draw uncertainty circle in degrees (~1 deg lat = 111 km)
        unc_deg = h.origin_uncertainty_km / 111.0
        unc_circle = patches.Circle(
            (h.estimated_origin_lon, h.estimated_origin_lat),
            unc_deg,
            edgecolor="#f85149",
            facecolor="#f85149",
            alpha=0.18,
            linestyle=":",
            linewidth=1.5,
            label=f"Source Uncertainty (±{h.origin_uncertainty_km:.1f} km)"
        )
        ax_hindcast.add_patch(unc_circle)

        ax_hindcast.set_xlabel("Longitude (°E)", color="#8b949e", fontsize=9)
        ax_hindcast.set_ylabel("Latitude (°N)", color="#8b949e", fontsize=9)
        ax_hindcast.tick_params(colors="#8b949e")
        ax_hindcast.legend(loc="upper left", facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

        # Hindcast metadata card
        h_info = (
            f"Model: {h.model_name}\n"
            f"Release Window: {h.origin_time_start.strftime('%H:%M')} - {h.origin_time_end.strftime('%H:%M UTC')}\n"
            f"Drift Duration: {h.drift_duration_hours:.1f} hrs | Uncertainty: ±{h.origin_uncertainty_km:.1f} km"
        )
        ax_hindcast.text(
            0.03, 0.05, h_info,
            color="#ffffff", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", fc="#0d1117", ec="#58a6ff", alpha=0.85),
            transform=ax_hindcast.transAxes
        )
    else:
        ax_hindcast.text(
            0.5, 0.5, "Drift Hindcasting Not Executed",
            color="#8b949e", ha="center", va="center", fontsize=11, transform=ax_hindcast.transAxes
        )

    # -------------------------------------------------------------
    # PANEL 4: AIS Vessel Traffic & Candidate Evidence
    # -------------------------------------------------------------
    ax_ais.set_facecolor("#161b22")
    for spine in ax_ais.spines.values():
        spine.set_color("#30363d")
    ax_ais.set_title("4. AIS Candidate Vessel Correlation & Evidence", color="#58a6ff", fontsize=12, fontweight="bold")

    if record.ais_available:
        if record.candidate_vessels:
            # Build horizontal table / breakdown of candidate vessels
            table_data = []
            col_labels = ["Rank", "Vessel / MMSI", "Type", "Closest (km)", "Time Off", "Score", "Compatibility"]
            for cand in record.candidate_vessels[:5]:
                m = cand.measurements
                v_name = m.get("ship_name") or f"{cand.mmsi}"
                v_type = m.get("ship_type") or "Unknown"
                d_km = f"{m.get('min_distance_km', 0.0):.1f}"
                dt_h = f"{m.get('time_offset_hours', 0.0):+.1f}h"
                score = f"{cand.overall_score:.2f}"
                cls_short = cand.classification.replace(" Candidate", "")
                table_data.append([f"#{cand.rank}", v_name[:14], v_type[:10], d_km, dt_h, score, cls_short])

            ax_ais.axis("off")
            tbl = ax_ais.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
                colColours=["#21262d"] * len(col_labels)
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1.0, 1.6)

            for key, cell in tbl.get_celld().items():
                cell.set_edgecolor("#30363d")
                if key[0] == 0:
                    cell.set_text_props(color="#58a6ff", fontweight="bold")
                else:
                    cell.set_facecolor("#161b22")
                    cell.set_text_props(color="#c9d1d9")

            # Scientific Disclaimer below table
            disclaimer = (
                "LEGAL & SCIENTIFIC DISCLAIMER:\n"
                "Candidate screening evaluates spatial-temporal transit compatibility only.\n"
                "Results do NOT constitute legal liability, proof of discharge, or fault attribution."
            )
            ax_ais.text(
                0.5, 0.08, disclaimer,
                color="#8b949e", ha="center", va="center", fontsize=7.5, style="italic",
                bbox=dict(boxstyle="round,pad=0.4", fc="#0d1117", ec="#8957e5", alpha=0.8),
                transform=ax_ais.transAxes
            )
        else:
            ax_ais.text(
                0.5, 0.5,
                f"AIS Checked: {record.total_vessels_checked} vessels.\nNo vessels found within {record.spill.origin_uncertainty_km:.0f} km search radius.",
                color="#8b949e", ha="center", va="center", fontsize=11, transform=ax_ais.transAxes
            )
    else:
        missing_reason = (
            "AIS Data Unavailable\n"
            "Source reconstruction completed successfully.\n"
            "Vessel correlation marked as unavailable (No synthetic/fabricated data)."
        )
        ax_ais.text(
            0.5, 0.5, missing_reason,
            color="#d29922", ha="center", va="center", fontsize=11, transform=ax_ais.transAxes
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=dpi)
        print(f"Investigation dashboard saved -> {output_path}")

    return fig


def generate_html_investigation_report(
    record: InvestigationRecord,
    dashboard_image_relpath: Optional[str] = None,
    output_html_path: Optional[str] = None,
) -> str:
    """
    Generates a modern, responsive, standalone HTML report for the investigation.
    """
    spill = record.spill
    la = record.look_alike_assessment
    h = record.hindcast_result

    # HTML builder
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OceanTrace Incident Investigation: {record.scene_id}</title>
  <style>
    :root {{
      --bg-dark: #0b111e;
      --card-bg: #161b22;
      --border-color: #30363d;
      --text-main: #c9d1d9;
      --text-bright: #f0f6fc;
      --accent-blue: #58a6ff;
      --accent-green: #3fb950;
      --accent-red: #f85149;
      --accent-amber: #d29922;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }}
    .header {{
      background: linear-gradient(135deg, #1f293d 0%, #161b22 100%);
      padding: 24px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      margin-bottom: 24px;
    }}
    .header h1 {{
      margin: 0 0 8px 0;
      color: var(--text-bright);
      font-size: 24px;
    }}
    .meta-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
      background-color: #21262d;
      border: 1px solid var(--border-color);
      margin-right: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background-color: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 20px;
    }}
    .card h2 {{
      color: var(--accent-blue);
      font-size: 16px;
      margin-top: 0;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 8px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      padding: 6px 0;
      border-bottom: 1px solid #21262d;
    }}
    .metric-label {{ color: #8b949e; font-size: 13px; }}
    .metric-val {{ color: var(--text-bright); font-weight: 600; font-size: 13px; }}
    .table-container {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }}
    th {{
      background-color: #21262d;
      color: var(--accent-blue);
    }}
    .disclaimer-box {{
      background-color: #161b22;
      border-left: 4px solid var(--accent-amber);
      padding: 16px;
      border-radius: 4px;
      margin-top: 24px;
      font-size: 13px;
      color: #8b949e;
    }}
    .dashboard-img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      margin-top: 20px;
    }}
  </style>
</head>
<body>

  <div class="header">
    <h1>OceanTrace Maritime Incident Investigation</h1>
    <div>
      <span class="meta-badge">Scene ID: {record.scene_id}</span>
      <span class="meta-badge">Investigated (UTC): {record.investigation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
      <span class="meta-badge" style="border-color: var(--accent-green); color: var(--accent-green);">Status: {('Detection Verified' if record.has_detection else 'No Detection')}</span>
    </div>
  </div>

  <div class="grid">
    <!-- Card 1: Spill Characterization -->
    <div class="card">
      <h2>1. Spill Characterization</h2>
      <div class="metric-row">
        <span class="metric-label">Observed Centroid</span>
        <span class="metric-val">{f'{spill.centroid_lat:.4f}°N, {spill.centroid_lon:.4f}°E' if spill else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Estimated Surface Area</span>
        <span class="metric-val">{f'{spill.area_km2:.3f} km²' if spill else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">SAR Observation Time</span>
        <span class="metric-val">{spill.observation_time.strftime('%Y-%m-%d %H:%M:%S UTC') if spill else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Polygon Boundaries</span>
        <span class="metric-val">{f'{len(spill.polygon)} vertices' if spill and spill.polygon else 'N/A'}</span>
      </div>
    </div>

    <!-- Card 2: Look-Alike Assessment -->
    <div class="card">
      <h2>2. Look-Alike & Ambiguity</h2>
      <div class="metric-row">
        <span class="metric-label">Classification</span>
        <span class="metric-val" style="color: {'var(--accent-red)' if la and la.is_look_alike_suspect else 'var(--accent-green)'}">
          {la.classification if la else 'Not Assessed'}
        </span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Ambiguity Score</span>
        <span class="metric-val">{f'{la.ambiguity_score:.2f} / 1.0' if la else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Damping Contrast</span>
        <span class="metric-val">{f'{la.damping_contrast_db:.2f} dB' if la and la.damping_contrast_db is not None else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Aspect Ratio (Morphology)</span>
        <span class="metric-val">{f'{la.aspect_ratio:.2f}' if la and la.aspect_ratio is not None else 'N/A'}</span>
      </div>
    </div>

    <!-- Card 3: Drift Hindcast -->
    <div class="card">
      <h2>3. Backward Drift Hindcasting</h2>
      <div class="metric-row">
        <span class="metric-label">Hindcasting Model</span>
        <span class="metric-val">{h.model_name if h else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Estimated Release Origin</span>
        <span class="metric-val">{f'{h.estimated_origin_lat:.4f}°N, {h.estimated_origin_lon:.4f}°E' if h else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Source Release Window</span>
        <span class="metric-val">{f'{h.origin_time_start.strftime("%H:%M")} - {h.origin_time_end.strftime("%H:%M UTC")}' if h else 'N/A'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Origin Uncertainty Radius</span>
        <span class="metric-val">{f'±{h.origin_uncertainty_km:.1f} km' if h else 'N/A'}</span>
      </div>
    </div>
  </div>

  <!-- Candidate Vessels Table -->
  <div class="card">
    <h2>4. AIS Candidate Vessel Correlation Evidence</h2>
    <div class="table-container">
"""

    if record.ais_available and record.candidate_vessels:
        html += """
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Vessel Name / MMSI</th>
            <th>Type</th>
            <th>Closest Approach (km)</th>
            <th>Time Offset (hrs)</th>
            <th>Compatibility Score</th>
            <th>Evidence Classification</th>
          </tr>
        </thead>
        <tbody>
"""
        for cand in record.candidate_vessels:
            m = cand.measurements
            v_name = m.get("ship_name") or f"MMSI {cand.mmsi}"
            v_type = m.get("ship_type") or "Unknown"
            d_km = f"{m.get('min_distance_km', 0.0):.2f}"
            dt_h = f"{m.get('time_offset_hours', 0.0):+.2f}"
            score = f"{cand.overall_score:.2f}"
            html += f"""
          <tr>
            <td><strong>#{cand.rank}</strong></td>
            <td>{v_name}</td>
            <td>{v_type}</td>
            <td>{d_km} km</td>
            <td>{dt_h} h</td>
            <td>{score}</td>
            <td><span class="meta-badge">{cand.classification}</span></td>
          </tr>
"""
        html += """
        </tbody>
      </table>
"""
    elif record.ais_available:
        html += f"<p>Zero vessels detected within origin uncertainty zone (Total broadcast tracks evaluated: {record.total_vessels_checked}).</p>"
    else:
        html += "<p style='color: var(--accent-amber);'>AIS broadcast data was not available for this observation region and time window. No candidate vessels correlated.</p>"

    html += """
    </div>
  </div>
"""

    if dashboard_image_relpath:
        html += f"""
  <div class="card">
    <h2>5. Visual Dashboard</h2>
    <img src="{dashboard_image_relpath}" alt="OceanTrace Investigation Dashboard" class="dashboard-img">
  </div>
"""

    html += """
  <div class="disclaimer-box">
    <strong>Scientific Integrity & Legal Standard:</strong>
    Candidate vessel screening results generated by OceanTrace represent spatial-temporal transit compatibility evidence for maritime safety screening only.
    Under scientific integrity guidelines, OceanTrace does NOT assert legal guilt, vessel fault, or regulatory responsibility.
  </div>

</body>
</html>
"""

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Investigation HTML report saved -> {output_html_path}")

    return html
