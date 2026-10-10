#!/usr/bin/env python3
"""
Stage 4B: Temporal Radar Return Association
===========================================
Project: Radar–Camera Fusion-Based Driver Assistance System (MY_ADAS_PROJECT)
Author / Pipeline: Stage 4B Temporal Radar Return Association Baseline

OBJECTIVE:
----------
Investigate whether radar returns from successive sweeps can be geometrically
associated as candidate observations of the same physical object under
ego-motion compensation.

IMPORTANT RESEARCH & SAFETY DISCLAIMER:
----------------------------------------
1. EXPLORATORY GEOMETRIC ASSOCIATION ONLY:
   This experiment implements an exploratory geometric association baseline.
   It is NOT a trained multi-object tracker (MOT), Kalman filter tracker, or
   learned temporal association network. Candidate associations must NOT be
   interpreted as confirmed vehicle tracks.
2. NO PERFORMANCE BENCHMARKING WITHOUT GROUND TRUTH:
   Tracking accuracy, precision, recall, MOTA, and identity consistency are NOT
   reported because ground-truth radar return correspondence labels are not
   available in nuScenes for individual radar reflections.
3. SCALAR RADAR DOPPLER INTERPRETATION & SIGN CONVENTION LIMITATIONS:
   Official devkit documentation (nuscenes/utils/data_classes.py) defines vx_comp simply as:
   "vx_comp, vy_comp are the velocities in m/s compensated by the ego motion" with "x is front, y is left".
   The devkit does NOT document an explicit sign convention (e.g. whether positive indicates approaching
   or receding). Therefore, neutral terminology ("compensated Doppler component") is used throughout.
   vx_comp represents a 1D scalar Doppler measurement along the radar line of sight (LOS) projected onto
   the sensor longitudinal axis. It does NOT represent a complete 2D Cartesian object velocity vector
   because transverse (cross-range) velocity is unmeasured by a standard Doppler radar beam. We do NOT
   propagate 2D positions using vx_comp/vy_comp.
   Crucially, when comparing returns across consecutive sweeps at different sensor line-of-sight angles
   (theta_1 != theta_2), the projected radial Doppler component changes purely due to geometric projection,
   even if the physical object maintains a constant Cartesian speed and heading. Conversely, distinct
   physical objects at different positions can coincidentally exhibit identical Doppler values.
   Therefore, Doppler gating (|dv| <= 2.5 m/s) is used strictly as an exploratory scalar consistency
   check to filter out contradictory Doppler states, NOT as 2D vector kinematics.
4. COORDINATE TRANSFORMATION vs. OBJECT TRACKING:
   Coordinate transformation into the reference ego vehicle frame (V_ego) compensates
   for ego-vehicle translation and yaw rotation. It aligns the coordinate axes,
   but does NOT track objects or establish physical identity across time.
5. EXPLORATORY GATING PARAMETERS:
   The 1.5 m spatial gate and 2.5 m/s Doppler-difference gate are exploratory
   assumptions chosen as heuristic baselines, NOT validated or standard tracking thresholds.
6. NO HAZARD OR COLLISION THRESHOLDS:
   No collision prediction, Time-to-Collision (TTC), emergency braking thresholds,
   or hazard conflict classifications are implemented.
"""

import os
import sys
import json
import csv
import argparse
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

# Import verified multi-sweep loader from Stage 4A to maintain absolute pipeline consistency
from stage4_radar_motion import load_multi_sweep_radar


def compute_consecutive_sweep_association(
    merged_data,
    sweep_records,
    src_sweep_idx=0,
    dst_sweep_idx=1,
    d_gate=1.5,
    dv_gate=2.5
):
    """
    Computes conservative temporal association between two consecutive radar sweeps.

    Parameters:
    -----------
    merged_data : dict
        Multi-sweep radar data in reference ego frame V_ego.
    sweep_records : list of dict
        Metadata for each sweep.
    src_sweep_idx : int
        Index of the earlier sweep (source).
    dst_sweep_idx : int
        Index of the subsequent sweep (destination).
    d_gate : float
        Exploratory spatial distance gate in meters (default: 1.5 m).
        Derived from an exploratory heuristic assuming max relative speed of 20 m/s
        over dt ≈ 75 ms. Explicitly an exploratory assumption; NOT a validated or standard tracking threshold.
    dv_gate : float
        Exploratory scalar Doppler difference gate in m/s (default: 2.5 m/s).
        Exploratory consistency check on the compensated Doppler component (vx_comp); NOT a validated tracking threshold.

    Returns:
    --------
    assoc_results : dict
        Detailed metrics, candidate pairs, ambiguous returns, unmatched returns, and partition integrity.
    """
    m_src = merged_data['sweep_idx'] == src_sweep_idx
    m_dst = merged_data['sweep_idx'] == dst_sweep_idx

    src_indices = np.where(m_src)[0]
    dst_indices = np.where(m_dst)[0]

    n_src = len(src_indices)
    n_dst = len(dst_indices)

    if n_src == 0 or n_dst == 0:
        raise ValueError(f"Empty point set encountered: n_src={n_src}, n_dst={n_dst}")

    # Extract coordinates in V_ego
    p_src = np.column_stack([merged_data['x'][src_indices], merged_data['y'][src_indices]])
    p_dst = np.column_stack([merged_data['x'][dst_indices], merged_data['y'][dst_indices]])

    v_src = merged_data['vx_comp'][src_indices]
    v_dst = merged_data['vx_comp'][dst_indices]

    id_src = merged_data['id'][src_indices]
    id_dst = merged_data['id'][dst_indices]

    dp_src = merged_data['dyn_prop'][src_indices]
    dp_dst = merged_data['dyn_prop'][dst_indices]

    t_src = sweep_records[src_sweep_idx]['timestamp']
    t_dst = sweep_records[dst_sweep_idx]['timestamp']
    delta_t_sec = (t_dst - t_src) / 1e6
    delta_t_ms = delta_t_sec * 1000.0

    # Pairwise 2D Euclidean distance matrix (N_src x N_dst)
    dist_matrix = np.linalg.norm(p_src[:, None, :] - p_dst[None, :, :], axis=-1)

    # Pairwise radial Doppler velocity difference matrix (N_src x N_dst)
    dv_matrix = np.abs(v_src[:, None] - v_dst[None, :])

    # Admissibility gate
    admissible = (dist_matrix <= d_gate) & (dv_matrix <= dv_gate)

    # Candidate lists
    src_candidates = [np.where(admissible[i, :])[0] for i in range(n_src)]
    dst_candidates = [np.where(admissible[:, j])[0] for j in range(n_dst)]

    unique_pairs = []
    ambiguous_src_local = set()
    ambiguous_dst_local = set()
    unmatched_src_local = set()
    unmatched_dst_local = set()

    for i in range(n_src):
        cands = src_candidates[i]
        if len(cands) == 0:
            unmatched_src_local.add(i)
        elif len(cands) == 1:
            j = cands[0]
            # Check if destination j also only points to source i
            if len(dst_candidates[j]) == 1:
                # Strictly isolated 1-to-1 match
                unique_pairs.append({
                    "src_local_idx": int(i),
                    "dst_local_idx": int(j),
                    "src_global_idx": int(src_indices[i]),
                    "dst_global_idx": int(dst_indices[j]),
                    "src_x": float(p_src[i, 0]),
                    "src_y": float(p_src[i, 1]),
                    "dst_x": float(p_dst[j, 0]),
                    "dst_y": float(p_dst[j, 1]),
                    "spatial_separation_m": float(dist_matrix[i, j]),
                    "src_vx_comp": float(v_src[i]),
                    "dst_vx_comp": float(v_dst[j]),
                    "delta_vx_comp": float(dv_matrix[i, j]),
                    "src_dyn_prop": int(dp_src[i]),
                    "dst_dyn_prop": int(dp_dst[j]),
                    "src_sensor_id": int(id_src[i]),
                    "dst_sensor_id": int(id_dst[j]),
                    "same_sensor_id": bool(id_src[i] == id_dst[j]),
                    "same_dyn_prop": bool(dp_src[i] == dp_dst[j])
                })
            else:
                # Destination j has multiple source candidates -> contention
                ambiguous_src_local.add(i)
                ambiguous_dst_local.add(j)
        else:
            # Source i has multiple destination candidates within gate
            ambiguous_src_local.add(i)
            for j in cands:
                ambiguous_dst_local.add(j)

    for j in range(n_dst):
        cands = dst_candidates[j]
        if len(cands) == 0:
            unmatched_dst_local.add(j)
        elif len(cands) > 1:
            ambiguous_dst_local.add(j)

    # Verification: Mutual exclusivity check
    # Every source return must belong to exactly one category
    matched_src_set = {p['src_local_idx'] for p in unique_pairs}
    assert len(matched_src_set.intersection(ambiguous_src_local)) == 0, "Overlap between unique and ambiguous src"
    assert len(matched_src_set.intersection(unmatched_src_local)) == 0, "Overlap between unique and unmatched src"
    assert len(ambiguous_src_local.intersection(unmatched_src_local)) == 0, "Overlap between ambiguous and unmatched src"
    src_partition_sum = len(matched_src_set) + len(ambiguous_src_local) + len(unmatched_src_local)
    assert src_partition_sum == n_src, "Source partition incomplete"

    matched_dst_set = {p['dst_local_idx'] for p in unique_pairs}
    assert len(matched_dst_set.intersection(ambiguous_dst_local)) == 0, "Overlap between unique and ambiguous dst"
    assert len(matched_dst_set.intersection(unmatched_dst_local)) == 0, "Overlap between unique and unmatched dst"
    assert len(ambiguous_dst_local.intersection(unmatched_dst_local)) == 0, "Overlap between ambiguous and unmatched dst"
    dst_partition_sum = len(matched_dst_set) + len(ambiguous_dst_local) + len(unmatched_dst_local)
    assert dst_partition_sum == n_dst, "Destination partition incomplete"

    # Separation statistics for unique pairs
    if len(unique_pairs) > 0:
        seps = [p['spatial_separation_m'] for p in unique_pairs]
        dvs = [p['delta_vx_comp'] for p in unique_pairs]
        same_id_cnt = sum(1 for p in unique_pairs if p['same_sensor_id'])
        same_dp_cnt = sum(1 for p in unique_pairs if p['same_dyn_prop'])
        sep_stats = {
            "mean_m": float(np.mean(seps)),
            "std_m": float(np.std(seps)),
            "min_m": float(np.min(seps)),
            "max_m": float(np.max(seps))
        }
        dv_stats = {
            "mean_m_s": float(np.mean(dvs)),
            "std_m_s": float(np.std(dvs)),
            "min_m_s": float(np.min(dvs)),
            "max_m_s": float(np.max(dvs))
        }
    else:
        sep_stats = {"mean_m": 0.0, "std_m": 0.0, "min_m": 0.0, "max_m": 0.0}
        dv_stats = {"mean_m_s": 0.0, "std_m_s": 0.0, "min_m_s": 0.0, "max_m_s": 0.0}
        same_id_cnt = 0
        same_dp_cnt = 0

    return {
        "src_sweep_idx": src_sweep_idx,
        "dst_sweep_idx": dst_sweep_idx,
        "src_timestamp": t_src,
        "dst_timestamp": t_dst,
        "delta_t_ms": round(float(delta_t_ms), 3),
        "total_source_returns": n_src,
        "total_destination_returns": n_dst,
        "d_gate_m": d_gate,
        "dv_gate_m_s": dv_gate,
        "gate_classification": "Exploratory assumptions (NOT validated or standard tracking thresholds)",
        "partition_integrity": {
            "source_partition_sum": src_partition_sum,
            "source_total_returns": n_src,
            "destination_partition_sum": dst_partition_sum,
            "destination_total_returns": n_dst,
            "double_counting_detected": False,
            "partitions_mutually_exclusive_and_complete": True
        },
        "unique_candidate_pairs_count": len(unique_pairs),
        "ambiguous_source_count": len(ambiguous_src_local),
        "ambiguous_destination_count": len(ambiguous_dst_local),
        "unmatched_source_count": len(unmatched_src_local),
        "unmatched_destination_count": len(unmatched_dst_local),
        "spatial_separation_stats_m": sep_stats,
        "doppler_difference_stats_m_s": dv_stats,
        "sensor_cluster_id_agreement_count": same_id_cnt,
        "dynamic_property_agreement_count": same_dp_cnt,
        "unique_pairs": unique_pairs,
        "ambiguous_src_indices": sorted(list(ambiguous_src_local)),
        "ambiguous_dst_indices": sorted(list(ambiguous_dst_local)),
        "unmatched_src_indices": sorted(list(unmatched_src_local)),
        "unmatched_dst_indices": sorted(list(unmatched_dst_local)),
        "p_src": p_src,
        "p_dst": p_dst
    }


def render_temporal_association_visualization(
    assoc_01,
    assoc_12,
    triplet_count,
    ref_info,
    output_filepath,
    show_plot=False
):
    """
    Renders a high-resolution dual-panel Bird's-Eye View (BEV) visualization:
    Panel 1: Sweep 0 -> Sweep 1 Temporal Association (dt = +75.03 ms)
    Panel 2: Sweep 1 -> Sweep 2 Temporal Association (dt = +75.18 ms)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=140, facecolor='#0d1117')
    plt.subplots_adjust(top=0.86, bottom=0.10, left=0.06, right=0.95, wspace=0.22)

    # Super Title with Explicit Methodology & Limitations
    fig.suptitle(
        "STAGE 4B: TEMPORAL RADAR RETURN ASSOCIATION (EXPLORATORY BASELINE)\n"
        "[Exploratory Spatial Gate (d <= 1.5 m) & Radial Doppler Gate (|dv| <= 2.5 m/s) -- NOT Confirmed Object Tracks]",
        fontsize=13,
        fontweight='bold',
        color='#58a6ff',
        y=0.96
    )

    panels = [
        (ax1, assoc_01, "Pair 1: Sweep 0 (Keyframe) -> Sweep 1 (+75.03 ms)", "Sweep 0", "Sweep 1", '#00f0ff', '#ffb454', 'o', 's'),
        (ax2, assoc_12, "Pair 2: Sweep 1 (+75.03 ms) -> Sweep 2 (+150.21 ms)", "Sweep 1", "Sweep 2", '#ffb454', '#d2a8ff', 's', '^')
    ]

    cone_dist = 55.0
    half_angle = np.deg2rad(35)

    for ax, assoc, panel_title, src_name, dst_name, src_col, dst_col, src_marker, dst_marker in panels:
        ax.set_facecolor('#0d1117')
        ax.set_aspect('equal')
        ax.set_xlim(-28, 28)
        ax.set_ylim(-3, 58)

        # 1. Coordinate Grid & Range Rings
        for r in [10, 20, 30, 40, 50]:
            circle = plt.Circle((0, 0), r, color='#30363d', fill=False, linestyle=':', linewidth=0.8, alpha=0.7)
            ax.add_patch(circle)
            ax.text(1.0, r + 0.4, f"{r}m", color='#6e7681', fontsize=7.5, ha='left')

        # Radar FOV Wedge
        fov_poly = Polygon(
            [[0, 0],
             [cone_dist * np.sin(-half_angle), cone_dist * np.cos(-half_angle)],
             [cone_dist * np.sin(half_angle), cone_dist * np.cos(half_angle)]],
            closed=True, facecolor='#161b22', edgecolor='#30363d', linestyle='--', linewidth=0.9, alpha=0.45
        )
        ax.add_patch(fov_poly)

        # 2. Ego Vehicle Glyph at Origin
        ego_box = plt.Rectangle((-1.0, -2.5), 2.0, 4.5, facecolor='#21262d', edgecolor='#58a6ff', linewidth=1.2, zorder=6)
        ax.add_patch(ego_box)
        ax.text(0, -0.2, "EGO", color='#58a6ff', fontsize=7, fontweight='bold', ha='center', va='center', zorder=7)

        p_src = assoc['p_src']
        p_dst = assoc['p_dst']

        # 3. Plot Unmatched Returns
        unmatched_src_idx = assoc['unmatched_src_indices']
        if len(unmatched_src_idx) > 0:
            ax.scatter(
                p_src[unmatched_src_idx, 1], p_src[unmatched_src_idx, 0],
                c='#484f58', s=35, marker=src_marker, edgecolors='#30363d', linewidths=0.8,
                alpha=0.6, zorder=3, label=f"Unmatched {src_name} ({len(unmatched_src_idx)})"
            )

        unmatched_dst_idx = assoc['unmatched_dst_indices']
        if len(unmatched_dst_idx) > 0:
            ax.scatter(
                p_dst[unmatched_dst_idx, 1], p_dst[unmatched_dst_idx, 0],
                c='#6e7681', s=35, marker=dst_marker, edgecolors='#30363d', linewidths=0.8,
                alpha=0.6, zorder=3, label=f"Unmatched {dst_name} ({len(unmatched_dst_idx)})"
            )

        # 4. Plot Ambiguous Returns
        ambig_src_idx = assoc['ambiguous_src_indices']
        if len(ambig_src_idx) > 0:
            ax.scatter(
                p_src[ambig_src_idx, 1], p_src[ambig_src_idx, 0],
                c='#f0883e', s=55, marker=src_marker, edgecolors='#ffa657', linewidths=1.2,
                alpha=0.85, zorder=4, label=f"Ambiguous {src_name} ({len(ambig_src_idx)})"
            )

        ambig_dst_idx = assoc['ambiguous_dst_indices']
        if len(ambig_dst_idx) > 0:
            ax.scatter(
                p_dst[ambig_dst_idx, 1], p_dst[ambig_dst_idx, 0],
                c='#d29922', s=55, marker=dst_marker, edgecolors='#f0883e', linewidths=1.2,
                alpha=0.85, zorder=4, label=f"Ambiguous {dst_name} ({len(ambig_dst_idx)})"
            )

        # 5. Plot Unique Candidate Associations & Linking Vectors
        pairs = assoc['unique_pairs']
        for pair in pairs:
            # Note: nuScenes frame x is longitudinal (forward), y is lateral (left)
            # In BEV plot: x-axis is lateral y, y-axis is longitudinal x
            sx, sy = pair['src_x'], pair['src_y']
            dx, dy = pair['dst_x'], pair['dst_y']

            # Vector line connecting source to destination
            ax.plot([sy, dy], [sx, dx], color='#3fb950', linewidth=1.5, alpha=0.9, zorder=5)

            # Source point
            ax.scatter(sy, sx, c=src_col, s=60, marker=src_marker, edgecolors='#ffffff', linewidths=0.9, zorder=6)
            # Destination point
            ax.scatter(dy, dx, c=dst_col, s=60, marker=dst_marker, edgecolors='#ffffff', linewidths=0.9, zorder=6)

        # Dummy scatter for legend
        ax.scatter([], [], c=src_col, s=55, marker=src_marker, edgecolors='#ffffff', label=f"Associated {src_name}")
        ax.scatter([], [], c=dst_col, s=55, marker=dst_marker, edgecolors='#ffffff', label=f"Associated {dst_name}")
        ax.plot([], [], color='#3fb950', linewidth=2.0, label=f"Candidate Association Link ({len(pairs)})")

        # Titles and Labels
        ax.set_title(panel_title, fontsize=11, fontweight='bold', color='#c9d1d9', pad=10)
        ax.set_xlabel("Lateral Position [m] (Left ← 0 → Right)", fontsize=9.5, color='#8b949e')
        ax.set_ylabel("Longitudinal Distance [m] (Forward)", fontsize=9.5, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        # Legend
        ax.legend(
            loc='upper right',
            fontsize=8,
            facecolor='#161b22',
            edgecolor='#30363d',
            labelcolor='#c9d1d9',
            framealpha=0.92
        )

        # Annotation Card with Empirical Numbers
        info_text = (
            f"Sweep dt: {assoc['delta_t_ms']:.1f} ms\n"
            f"Exploratory Gate: d<={assoc['d_gate_m']}m, |dv|<={assoc['dv_gate_m_s']}m/s\n"
            f"[Exploratory Heuristic -- Not Standard Thresholds]\n"
            f"--------------------------\n"
            f"Source Returns : {assoc['total_source_returns']}\n"
            f"Dest Returns   : {assoc['total_destination_returns']}\n"
            f"--------------------------\n"
            f"Unique Pairs   : {assoc['unique_candidate_pairs_count']}\n"
            f"Ambiguous Src  : {assoc['ambiguous_source_count']}\n"
            f"Ambiguous Dst  : {assoc['ambiguous_destination_count']}\n"
            f"Unmatched Src  : {assoc['unmatched_source_count']}\n"
            f"Unmatched Dst  : {assoc['unmatched_destination_count']}\n"
            f"--------------------------\n"
            f"Mean Sep       : {assoc['spatial_separation_stats_m']['mean_m']:.2f} m\n"
            f"Mean |dv|      : {assoc['doppler_difference_stats_m_s']['mean_m_s']:.2f} m/s\n"
            f"Cluster ID Mtch: {assoc['sensor_cluster_id_agreement_count']}"
        )
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=8.0,
            family='monospace',
            color='#c9d1d9',
            va='top',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', edgecolor='#30363d', alpha=0.92)
        )

    # Global Bottom Note
    bottom_note = (
        f"EXPLORATORY BASELINE SUMMARY: Found {assoc_01['unique_candidate_pairs_count']} candidate pairs (Sweep 0->1) "
        f"and {assoc_12['unique_candidate_pairs_count']} candidate pairs (Sweep 1->2); {triplet_count} candidate persistence chains across 3 sweeps. "
        "Candidate associations are exploratory geometric proximity links under ego-motion compensation. "
        "NOT confirmed vehicle tracks. No collision prediction or hazard classification implied."
    )
    fig.text(0.5, 0.025, bottom_note, ha='center', fontsize=8.5, color='#8b949e', style='italic')

    # Save artifact
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=140, facecolor=fig.get_facecolor(), bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)


def export_candidate_pairs_csv(assoc_01, assoc_12, csv_filepath):
    """
    Exports a structured CSV of all candidate return associations.
    """
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    fieldnames = [
        "pair_label",
        "src_sweep_idx",
        "dst_sweep_idx",
        "src_local_idx",
        "dst_local_idx",
        "src_global_idx",
        "dst_global_idx",
        "delta_t_ms",
        "src_x_ego_m",
        "src_y_ego_m",
        "dst_x_ego_m",
        "dst_y_ego_m",
        "spatial_separation_m",
        "src_vx_comp_m_s",
        "dst_vx_comp_m_s",
        "delta_vx_comp_m_s",
        "src_dyn_prop",
        "dst_dyn_prop",
        "src_sensor_id",
        "dst_sensor_id",
        "same_sensor_id",
        "same_dyn_prop",
        "association_status"
    ]

    with open(csv_filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for p in assoc_01['unique_pairs']:
            writer.writerow({
                "pair_label": "sweep0_to_sweep1",
                "src_sweep_idx": assoc_01['src_sweep_idx'],
                "dst_sweep_idx": assoc_01['dst_sweep_idx'],
                "src_local_idx": p['src_local_idx'],
                "dst_local_idx": p['dst_local_idx'],
                "src_global_idx": p['src_global_idx'],
                "dst_global_idx": p['dst_global_idx'],
                "delta_t_ms": assoc_01['delta_t_ms'],
                "src_x_ego_m": round(p['src_x'], 3),
                "src_y_ego_m": round(p['src_y'], 3),
                "dst_x_ego_m": round(p['dst_x'], 3),
                "dst_y_ego_m": round(p['dst_y'], 3),
                "spatial_separation_m": round(p['spatial_separation_m'], 3),
                "src_vx_comp_m_s": round(p['src_vx_comp'], 3),
                "dst_vx_comp_m_s": round(p['dst_vx_comp'], 3),
                "delta_vx_comp_m_s": round(p['delta_vx_comp'], 3),
                "src_dyn_prop": p['src_dyn_prop'],
                "dst_dyn_prop": p['dst_dyn_prop'],
                "src_sensor_id": p['src_sensor_id'],
                "dst_sensor_id": p['dst_sensor_id'],
                "same_sensor_id": p['same_sensor_id'],
                "same_dyn_prop": p['same_dyn_prop'],
                "association_status": "unique_candidate_pair"
            })

        for p in assoc_12['unique_pairs']:
            writer.writerow({
                "pair_label": "sweep1_to_sweep2",
                "src_sweep_idx": assoc_12['src_sweep_idx'],
                "dst_sweep_idx": assoc_12['dst_sweep_idx'],
                "src_local_idx": p['src_local_idx'],
                "dst_local_idx": p['dst_local_idx'],
                "src_global_idx": p['src_global_idx'],
                "dst_global_idx": p['dst_global_idx'],
                "delta_t_ms": assoc_12['delta_t_ms'],
                "src_x_ego_m": round(p['src_x'], 3),
                "src_y_ego_m": round(p['src_y'], 3),
                "dst_x_ego_m": round(p['dst_x'], 3),
                "dst_y_ego_m": round(p['dst_y'], 3),
                "spatial_separation_m": round(p['spatial_separation_m'], 3),
                "src_vx_comp_m_s": round(p['src_vx_comp'], 3),
                "dst_vx_comp_m_s": round(p['dst_vx_comp'], 3),
                "delta_vx_comp_m_s": round(p['delta_vx_comp'], 3),
                "src_dyn_prop": p['src_dyn_prop'],
                "dst_dyn_prop": p['dst_dyn_prop'],
                "src_sensor_id": p['src_sensor_id'],
                "dst_sensor_id": p['dst_sensor_id'],
                "same_sensor_id": p['same_sensor_id'],
                "same_dyn_prop": p['same_dyn_prop'],
                "association_status": "unique_candidate_pair"
            })


def run_stage4b_temporal_association(
    sample_idx=0,
    nsweeps=3,
    d_gate=1.5,
    dv_gate=2.5,
    output_dir="outputs/temporal_association",
    show_plot=False
):
    """
    Executes Stage 4B Temporal Radar Return Association.
    """
    print("=================================================================")
    print(" STAGE 4B: TEMPORAL RADAR RETURN ASSOCIATION BASELINE")
    print("=================================================================")
    print(f"[*] Initializing nuScenes v1.0-mini from root directory...")
    nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)

    if sample_idx >= len(nusc.sample):
        raise IndexError(f"Sample index {sample_idx} exceeds dataset size ({len(nusc.sample)}).")

    sample = nusc.sample[sample_idx]
    sample_token = sample['token']
    scene_record = nusc.get('scene', sample['scene_token'])
    scene_name = scene_record['name']

    print(f"[*] Target Sample Token     : {sample_token}")
    print(f"[*] Scene                   : {scene_name}")
    print(f"[*] Accumulating {nsweeps} consecutive sweeps into V_ego reference frame...")

    merged_data, sweep_records, ref_info = load_multi_sweep_radar(nusc, sample, nsweeps=nsweeps)

    # 1. Compute Pairwise Temporal Associations
    print(f"[*] Evaluating Pair 1: Sweep 0 -> Sweep 1 (Gate: d<={d_gate}m, dv<={dv_gate}m/s)...")
    assoc_01 = compute_consecutive_sweep_association(
        merged_data=merged_data,
        sweep_records=sweep_records,
        src_sweep_idx=0,
        dst_sweep_idx=1,
        d_gate=d_gate,
        dv_gate=dv_gate
    )

    print(f"[*] Evaluating Pair 2: Sweep 1 -> Sweep 2 (Gate: d<={d_gate}m, dv<={dv_gate}m/s)...")
    assoc_12 = compute_consecutive_sweep_association(
        merged_data=merged_data,
        sweep_records=sweep_records,
        src_sweep_idx=1,
        dst_sweep_idx=2,
        d_gate=d_gate,
        dv_gate=dv_gate
    )

    # 2. Check 3-Sweep Persistent Triplet Chains
    # Link: Sweep 0 (i) -> Sweep 1 (j) AND Sweep 1 (j) -> Sweep 2 (k)
    map_01_dst_to_src = {p['dst_local_idx']: p for p in assoc_01['unique_pairs']}
    map_12_src_to_dst = {p['src_local_idx']: p for p in assoc_12['unique_pairs']}

    triplet_chains = []
    for mid_j in map_01_dst_to_src:
        if mid_j in map_12_src_to_dst:
            p01 = map_01_dst_to_src[mid_j]
            p12 = map_12_src_to_dst[mid_j]
            triplet_chains.append({
                "sweep0_local_idx": p01['src_local_idx'],
                "sweep1_local_idx": mid_j,
                "sweep2_local_idx": p12['dst_local_idx'],
                "sep_01_m": p01['spatial_separation_m'],
                "sep_12_m": p12['spatial_separation_m'],
                "mean_sep_m": (p01['spatial_separation_m'] + p12['spatial_separation_m']) / 2.0
            })
    triplet_count = len(triplet_chains)

    # 3. Print Quantitative Console Summary
    print("-----------------------------------------------------------------")
    print(" TEMPORAL RADAR ASSOCIATION QUANTITATIVE SUMMARY:")
    print("-----------------------------------------------------------------")
    print(f" Reference Coordinate Frame    : {ref_info['ref_coordinate_frame']}")
    print(f" Gating Criteria (Exploratory) : d <= {d_gate} m, |dv_comp| <= {dv_gate} m/s")
    print("   -> Status: Exploratory baseline assumptions (NOT validated or standard tracking thresholds)")
    print("-----------------------------------------------------------------")
    print(f" PAIR 1 (Sweep 0 -> Sweep 1, dt = {assoc_01['delta_t_ms']:+.2f} ms):")
    print(f"   -> Total Source Returns     : {assoc_01['total_source_returns']} (Sweep 0)")
    print(f"   -> Total Destination Returns: {assoc_01['total_destination_returns']} (Sweep 1)")
    print(f"   -> Unique Candidate Pairs   : {assoc_01['unique_candidate_pairs_count']} pairs")
    print(f"   -> Ambiguous Source Returns : {assoc_01['ambiguous_source_count']} returns (competing candidates)")
    print(f"   -> Ambiguous Dest Returns   : {assoc_01['ambiguous_destination_count']} returns (contested candidates)")
    print(f"   -> Unmatched Source Returns : {assoc_01['unmatched_source_count']} returns (0 candidates within gate)")
    print(f"   -> Unmatched Dest Returns   : {assoc_01['unmatched_destination_count']} returns (0 candidates within gate)")
    print(f"   -> Partition Check (Src)    : {assoc_01['unique_candidate_pairs_count']} unique + {assoc_01['ambiguous_source_count']} ambig + {assoc_01['unmatched_source_count']} unmatched = {assoc_01['partition_integrity']['source_partition_sum']} / {assoc_01['total_source_returns']} (100%, no double-counting)")
    print(f"   -> Partition Check (Dst)    : {assoc_01['unique_candidate_pairs_count']} unique + {assoc_01['ambiguous_destination_count']} ambig + {assoc_01['unmatched_destination_count']} unmatched = {assoc_01['partition_integrity']['destination_partition_sum']} / {assoc_01['total_destination_returns']} (100%, no double-counting)")
    print(f"   -> Spatial Separation       : mean={assoc_01['spatial_separation_stats_m']['mean_m']:.3f} m, max={assoc_01['spatial_separation_stats_m']['max_m']:.3f} m")
    print(f"   -> Doppler Velocity Diff    : mean={assoc_01['doppler_difference_stats_m_s']['mean_m_s']:.3f} m/s, max={assoc_01['doppler_difference_stats_m_s']['max_m_s']:.3f} m/s")
    print(f"   -> Same Sensor Cluster ID   : {assoc_01['sensor_cluster_id_agreement_count']} / {assoc_01['unique_candidate_pairs_count']} pairs")
    print(f"   -> Same Dynamic Property    : {assoc_01['dynamic_property_agreement_count']} / {assoc_01['unique_candidate_pairs_count']} pairs")
    print("-----------------------------------------------------------------")
    print(f" PAIR 2 (Sweep 1 -> Sweep 2, dt = {assoc_12['delta_t_ms']:+.2f} ms):")
    print(f"   -> Total Source Returns     : {assoc_12['total_source_returns']} (Sweep 1)")
    print(f"   -> Total Destination Returns: {assoc_12['total_destination_returns']} (Sweep 2)")
    print(f"   -> Unique Candidate Pairs   : {assoc_12['unique_candidate_pairs_count']} pairs")
    print(f"   -> Ambiguous Source Returns : {assoc_12['ambiguous_source_count']} returns (competing candidates)")
    print(f"   -> Ambiguous Dest Returns   : {assoc_12['ambiguous_destination_count']} returns (contested candidates)")
    print(f"   -> Unmatched Source Returns : {assoc_12['unmatched_source_count']} returns (0 candidates within gate)")
    print(f"   -> Unmatched Dest Returns   : {assoc_12['unmatched_destination_count']} returns (0 candidates within gate)")
    print(f"   -> Partition Check (Src)    : {assoc_12['unique_candidate_pairs_count']} unique + {assoc_12['ambiguous_source_count']} ambig + {assoc_12['unmatched_source_count']} unmatched = {assoc_12['partition_integrity']['source_partition_sum']} / {assoc_12['total_source_returns']} (100%, no double-counting)")
    print(f"   -> Partition Check (Dst)    : {assoc_12['unique_candidate_pairs_count']} unique + {assoc_12['ambiguous_destination_count']} ambig + {assoc_12['unmatched_destination_count']} unmatched = {assoc_12['partition_integrity']['destination_partition_sum']} / {assoc_12['total_destination_returns']} (100%, no double-counting)")
    print(f"   -> Spatial Separation       : mean={assoc_12['spatial_separation_stats_m']['mean_m']:.3f} m, max={assoc_12['spatial_separation_stats_m']['max_m']:.3f} m")
    print(f"   -> Doppler Velocity Diff    : mean={assoc_12['doppler_difference_stats_m_s']['mean_m_s']:.3f} m/s, max={assoc_12['doppler_difference_stats_m_s']['max_m_s']:.3f} m/s")
    print(f"   -> Same Sensor Cluster ID   : {assoc_12['sensor_cluster_id_agreement_count']} / {assoc_12['unique_candidate_pairs_count']} pairs")
    print(f"   -> Same Dynamic Property    : {assoc_12['dynamic_property_agreement_count']} / {assoc_12['unique_candidate_pairs_count']} pairs")
    print("-----------------------------------------------------------------")
    print(" MULTI-SWEEP CANDIDATE PERSISTENCE:")
    print(f"   -> Candidate Persistence Chains: {triplet_count} candidate association chains across all 3 sweeps (0 -> 1 -> 2)")
    print("   -> Explicit Status: Candidate persistence chains; NOT confirmed physical-object tracks.")
    print("-----------------------------------------------------------------")
    print(" DOCUMENTED ASSUMPTIONS & LIMITATIONS:")
    print(" 1. Exploratory Geometric Association: Candidate associations are proximity links")
    print("    under ego-motion compensation. They are NOT certified vehicle tracks.")
    print(" 2. Exploratory Gating Assumptions: The 1.5 m spatial gate and 2.5 m/s Doppler gate are")
    print("    preliminary heuristics, NOT validated or standard multi-object tracking thresholds.")
    print(" 3. 1D Scalar Doppler & Sensor Line-of-Sight Effect: vx_comp is the ego-compensated Doppler")
    print("    component projected onto the sensor x-axis. Official devkit (nuscenes/utils/data_classes.py)")
    print("    defines vx_comp as 'velocities in m/s compensated by ego motion' but does NOT document an explicit")
    print("    sign convention (approaching vs receding); neutral wording is maintained. Transverse velocity is")
    print("    unmeasured. Returns at different azimuth angles (theta_1 != theta_2) exhibit different projected")
    print("    Doppler components even for constant Cartesian velocity. 2D kinematic extrapolation is not valid.")
    print(" 4. No Tracking Performance Claims: Precision, recall, and identity switches cannot")
    print("    be reported without ground truth radar tracklet associations.")
    print(" 5. Radar Firmware Cluster ID: Cluster IDs are recycled sensor table slots (0-127);")
    print("    they do not serve as reliable multi-sweep tracklet identifiers.")
    print(" 6. Non-Predictive Scope: No collision prediction, TTC, or hazard zones are evaluated.")
    print("-----------------------------------------------------------------")

    # 4. Export JSON Metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "temporal_association_metrics.json")
    metrics_data = {
        "pipeline_stage": "Stage 4B Temporal Radar Return Association",
        "sample_token": sample_token,
        "scene_name": scene_name,
        "reference_coordinate_frame": ref_info['ref_coordinate_frame'],
        "nsweeps_evaluated": nsweeps,
        "filter_settings": {
            "nuScenes_devkit_filters": {
                "invalid_states": "[0] (valid cluster)",
                "dynprop_states": "range(7) (moving, stationary, etc.)",
                "ambig_states": "[3] (unambiguous Doppler solution)"
            },
            "downstream_custom_filters": "none"
        },
        "association_parameters": {
            "spatial_distance_gate_m": d_gate,
            "radial_doppler_gate_m_s": dv_gate,
            "gate_classification": "Exploratory assumptions (NOT validated or standard tracking thresholds)",
            "methodology": "Conservative spatial gating with Doppler consistency and strict mutual 1-to-1 isolation",
            "justification_and_limitations": (
                "The 1.5 m spatial distance threshold and 2.5 m/s Doppler-difference threshold are exploratory "
                "assumptions derived from preliminary heuristics (assuming max relative speed <= 20 m/s over "
                "dt ≈ 75 ms). They are explicitly exploratory assumptions and are NOT claimed as validated, "
                "certified, or standard multi-object tracking thresholds."
            )
        },
        "doppler_velocity_interpretation_and_angle_limitations": {
            "field_used": "vx_comp (nuScenes point cloud row 8)",
            "devkit_definition": "vx_comp, vy_comp are the velocities in m/s compensated by the ego motion (nuscenes/utils/data_classes.py; x is front, y is left).",
            "sign_convention_status": "Unverified in official devkit documentation (no explicit definition of approaching vs receding sign); neutral terminology 'compensated Doppler component' is strictly maintained.",
            "physical_interpretation": "1D scalar Doppler measurement along the radar line of sight projected onto the sensor x-axis.",
            "measurement_nature": "Scalar line-of-sight velocity measurement only, NOT a complete 2D or 3D Cartesian velocity vector.",
            "sensor_angle_dependency": (
                "Because radial Doppler measures velocity along the sensor line-of-sight ray, returns at different "
                "azimuth angles (theta_1 != theta_2) have different projected Doppler values even if the target vehicle "
                "maintains constant Cartesian speed and heading. Conversely, separate objects at different bearings can "
                "coincidentally have identical Doppler values."
            ),
            "unmeasured_components": (
                "Transverse (cross-range) velocity is fundamentally unmeasured by a single Doppler radar beam. "
                "Kinematic forward-projection using vx_comp as a Cartesian velocity vector is physically invalid and is not performed."
            )
        },
        "consecutive_pairs": {
            "sweep_0_to_1": {
                "source_sweep_index": assoc_01['src_sweep_idx'],
                "destination_sweep_index": assoc_01['dst_sweep_idx'],
                "source_timestamp": assoc_01['src_timestamp'],
                "destination_timestamp": assoc_01['dst_timestamp'],
                "delta_t_ms": assoc_01['delta_t_ms'],
                "source_returns_count": assoc_01['total_source_returns'],
                "destination_returns_count": assoc_01['total_destination_returns'],
                "unique_candidate_pairs_count": assoc_01['unique_candidate_pairs_count'],
                "ambiguous_source_count": assoc_01['ambiguous_source_count'],
                "ambiguous_destination_count": assoc_01['ambiguous_destination_count'],
                "unmatched_source_count": assoc_01['unmatched_source_count'],
                "unmatched_destination_count": assoc_01['unmatched_destination_count'],
                "partition_integrity": assoc_01['partition_integrity'],
                "spatial_separation_stats_m": assoc_01['spatial_separation_stats_m'],
                "doppler_difference_stats_m_s": assoc_01['doppler_difference_stats_m_s'],
                "sensor_cluster_id_agreement_count": assoc_01['sensor_cluster_id_agreement_count'],
                "dynamic_property_agreement_count": assoc_01['dynamic_property_agreement_count']
            },
            "sweep_1_to_2": {
                "source_sweep_index": assoc_12['src_sweep_idx'],
                "destination_sweep_index": assoc_12['dst_sweep_idx'],
                "source_timestamp": assoc_12['src_timestamp'],
                "destination_timestamp": assoc_12['dst_timestamp'],
                "delta_t_ms": assoc_12['delta_t_ms'],
                "source_returns_count": assoc_12['total_source_returns'],
                "destination_returns_count": assoc_12['total_destination_returns'],
                "unique_candidate_pairs_count": assoc_12['unique_candidate_pairs_count'],
                "ambiguous_source_count": assoc_12['ambiguous_source_count'],
                "ambiguous_destination_count": assoc_12['ambiguous_destination_count'],
                "unmatched_source_count": assoc_12['unmatched_source_count'],
                "unmatched_destination_count": assoc_12['unmatched_destination_count'],
                "partition_integrity": assoc_12['partition_integrity'],
                "spatial_separation_stats_m": assoc_12['spatial_separation_stats_m'],
                "doppler_difference_stats_m_s": assoc_12['doppler_difference_stats_m_s'],
                "sensor_cluster_id_agreement_count": assoc_12['sensor_cluster_id_agreement_count'],
                "dynamic_property_agreement_count": assoc_12['dynamic_property_agreement_count']
            }
        },
        "multi_sweep_persistence": {
            "candidate_persistence_chains_count": triplet_count,
            "description": "Candidate persistence chains maintaining unique 1-to-1 candidate associations continuously across all 3 sweeps (0 -> 1 -> 2); exploratory geometric links, NOT confirmed physical-object tracks."
        },
        "documented_limitations": [
            "Candidate associations are exploratory geometric links and are NOT verified vehicle tracks.",
            "The 1.5 m distance gate and 2.5 m/s Doppler gate are exploratory assumptions, NOT validated or standard tracking thresholds.",
            "Radar Doppler velocity (vx_comp) is a 1D scalar compensated Doppler component; sign convention is not defined in official devkit documentation, transverse velocity is unmeasured, and 2D kinematic extrapolation is not valid.",
            "Returns at different sensor line-of-sight angles exhibit different projected Doppler values even for constant vehicle velocity.",
            "No precision, recall, MOTA, or identity consistency tracking performance metrics can be claimed without ground-truth return correspondence.",
            "Radar cluster IDs are recycled hardware table slots (0-127) and do not represent persistent global track IDs.",
            "Coordinate transformation into the reference ego frame compensates ego-motion, but does not compensate object motion or perform object tracking.",
            "No collision prediction, crash probability, time-to-collision (TTC), or hazard thresholds are implemented."
        ]
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[*] Association metrics saved : {os.path.abspath(metrics_path)}")

    # 5. Export Candidate Pairs CSV
    csv_path = os.path.join(output_dir, "temporal_association_candidates.csv")
    export_candidate_pairs_csv(assoc_01, assoc_12, csv_path)
    print(f"[*] Candidate pairs CSV saved : {os.path.abspath(csv_path)}")

    # 6. Render Visualisation
    vis_path = os.path.join(output_dir, "temporal_association.png")
    print("[*] Generating temporal association BEV visualisation...")
    render_temporal_association_visualization(
        assoc_01=assoc_01,
        assoc_12=assoc_12,
        triplet_count=triplet_count,
        ref_info=ref_info,
        output_filepath=vis_path,
        show_plot=show_plot
    )
    print(f"[*] Visualisation saved       : {os.path.abspath(vis_path)}")
    print("=================================================================")
    print(" STAGE 4B TEMPORAL ASSOCIATION COMPLETED SUCCESSFULLY.")
    print("=================================================================")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4B: Temporal Radar Return Association")
    parser.add_argument("--sample-idx", type=int, default=0, help="Target keyframe sample index (default: 0)")
    parser.add_argument("--nsweeps", type=int, default=3, help="Number of sweeps to accumulate (default: 3)")
    parser.add_argument("--d-gate", type=float, default=1.5, help="Exploratory spatial distance gate in meters (default: 1.5; NOT a validated tracking threshold)")
    parser.add_argument("--dv-gate", type=float, default=2.5, help="Exploratory Doppler velocity gate in m/s (default: 2.5; NOT a validated tracking threshold)")
    parser.add_argument("--output-dir", type=str, default="outputs/temporal_association", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display figure window")

    args = parser.parse_args()

    run_stage4b_temporal_association(
        sample_idx=args.sample_idx,
        nsweeps=args.nsweeps,
        d_gate=args.d_gate,
        dv_gate=args.dv_gate,
        output_dir=args.output_dir,
        show_plot=args.show
    )
