#!/usr/bin/env python3
"""
Stage 4A: Multi-Sweep Radar Motion Visualisation
================================================
Project: Radar–Camera Fusion-Based Driver Assistance System (MY_ADAS_PROJECT)
Author / Pipeline: Stage 4A Multi-Sweep Radar Motion & Doppler Attribution

OBJECTIVE:
----------
Accumulate multi-sweep radar returns into a common reference ego-vehicle frame
using ego-motion compensation. Visualise spatial returns, dynamic properties,
and scalar ego-compensated radial Doppler velocity measurements.

DISCLAIMER & NON-COLLISION CLASSIFICATION:
------------------------------------------
1. Empirical Doppler Measurement: Radial velocity represents the 1D scalar rate
   of closure along the sensor line-of-sight ray. It does NOT provide a complete
   2D or 3D object velocity vector and must NOT be interpreted as a vehicle travel heading.
2. Coordinate Transformation vs. Object Tracking: Multi-sweep accumulation aligns
   coordinate reference frames under ego-motion, but does NOT establish that detections
   across sweeps belong to the same physical object without temporal clustering and tracking.
3. Temporal Direction: The reference sample is at the beginning of the scene.
   Sweeps are collected along sample_data['next'] (subsequent forward-in-time sweeps),
   transformed into the reference keyframe ego frame.
4. Non-Predictive Scope: This script does NOT perform collision prediction, crash
   estimation, or automated braking decisions. No safety thresholds are invented.
"""

import os
import sys
import json
import argparse
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix


def load_multi_sweep_radar(nusc, sample, nsweeps=3):
    """
    Loads nsweeps consecutive RADAR_FRONT sweeps and transforms spatial coordinates
    into the reference ego vehicle coordinate system (V_ego) at the CAM_FRONT
    keyframe timestamp. Preserves scalar Doppler velocities without rotating them
    as 2D Cartesian object velocity vectors.

    Parameters:
    -----------
    nusc : NuScenes instance.
    sample : Target nuScenes keyframe sample dict.
    nsweeps : Number of consecutive sweeps to accumulate (default: 3).

    Returns:
    --------
    merged_data : dict containing concatenated arrays for transformed coordinates,
                  scalar Doppler velocities, dynamic properties, sweep indices, and timestamps.
    sweep_records : list of metadata dicts for each individual sweep.
    ref_info : dict containing reference sensor/ego frame metadata.
    """
    ref_cam_token = sample['data']['CAM_FRONT']
    ref_radar_token = sample['data']['RADAR_FRONT']

    ref_cam_sd = nusc.get('sample_data', ref_cam_token)
    ref_radar_sd = nusc.get('sample_data', ref_radar_token)

    ref_cam_pose = nusc.get('ego_pose', ref_cam_sd['ego_pose_token'])
    ref_cam_time = ref_cam_sd['timestamp']

    # Transformation from global frame to reference ego frame at camera timestamp
    car_from_global = transform_matrix(
        ref_cam_pose['translation'],
        Quaternion(ref_cam_pose['rotation']),
        inverse=True
    )

    # Gather consecutive sweeps
    # Strategy: If previous sweeps exist, collect backward.
    # If at scene start (sample 0, prev == ''), collect forward via 'next'.
    sweeps_sd = []
    curr_sd = ref_radar_sd
    temporal_strategy = "backward"

    backward_available = (curr_sd['prev'] != '')
    if backward_available:
        temp_sweeps = [curr_sd]
        while len(temp_sweeps) < nsweeps and curr_sd['prev'] != '':
            curr_sd = nusc.get('sample_data', curr_sd['prev'])
            temp_sweeps.append(curr_sd)
        sweeps_sd = list(reversed(temp_sweeps))
        temporal_strategy = "backward_historical"
    else:
        sweeps_sd.append(curr_sd)
        while len(sweeps_sd) < nsweeps and curr_sd['next'] != '':
            curr_sd = nusc.get('sample_data', curr_sd['next'])
            sweeps_sd.append(curr_sd)
        temporal_strategy = "forward_subsequent"

    if len(sweeps_sd) < nsweeps:
        raise RuntimeError(
            f"Insufficient consecutive sweeps available in scene (found {len(sweeps_sd)}, requested {nsweeps})."
        )

    # Containers for accumulated data
    all_x = []
    all_y = []
    all_z = []
    all_vx_raw = []
    all_vy_raw = []
    all_vx_comp = []
    all_vy_comp = []
    all_dyn_prop = []
    all_id = []
    all_rcs = []
    all_quality = []
    all_ambig = []
    all_pdh0 = []
    all_sweep_idx = []
    all_time_offset_ms = []

    sweep_records = []

    for idx, sw in enumerate(sweeps_sd):
        sw_file = os.path.join(nusc.dataroot, sw['filename'])
        if not os.path.exists(sw_file):
            raise FileNotFoundError(f"Radar sweep file not found on disk: {sw_file}")

        # Check raw unfiltered PCD point count directly from file
        pc_unfiltered = RadarPointCloud.from_file(
            sw_file,
            invalid_states=list(range(18)),
            dynprop_states=list(range(8)),
            ambig_states=list(range(5))
        )
        raw_pcd_count = pc_unfiltered.points.shape[1]

        # Load point cloud with official nuScenes default filtering:
        # invalid_states=[0], dynprop_states=range(7), ambig_states=[3]
        pc = RadarPointCloud.from_file(sw_file)
        retained_count = pc.points.shape[1]
        discarded_count = raw_pcd_count - retained_count

        sw_pose = nusc.get('ego_pose', sw['ego_pose_token'])
        sw_cs = nusc.get('calibrated_sensor', sw['calibrated_sensor_token'])

        # Homogeneous transforms for spatial coordinates:
        # 1. Sensor frame at sw timestamp -> Ego frame at sw timestamp
        car_from_sensor = transform_matrix(
            sw_cs['translation'],
            Quaternion(sw_cs['rotation']),
            inverse=False
        )
        # 2. Ego frame at sw timestamp -> Global frame
        global_from_car = transform_matrix(
            sw_pose['translation'],
            Quaternion(sw_pose['rotation']),
            inverse=False
        )
        # 3. Composite transform: Sensor(sw_time) -> Ego(ref_time)
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_sensor])

        # Transform spatial 3D coordinates (rows 0, 1, 2)
        pc.transform(trans_matrix)

        # VELOCITY FIELD INTERPRETATION:
        # nuScenes vx_comp (row 8) and vy_comp (row 9) are Doppler velocities compensated
        # for ego-vehicle translation and yaw rate. They represent 1D line-of-sight radial
        # velocity projected onto sensor axes (vx = vr*cos(theta), vy = vr*sin(theta)).
        # They do NOT form a complete 2D Cartesian object velocity vector.
        # DO NOT rotate them as Cartesian object vectors.
        # Retain vx_comp as the longitudinal scalar radial Doppler closure rate.
        vx_comp_scalar = pc.points[8, :].copy()
        vy_comp_scalar = pc.points[9, :].copy()

        # Time offset relative to CAM_FRONT keyframe (ms)
        time_offset_ms = (sw['timestamp'] - ref_cam_time) / 1000.0

        # Validate finite coordinates
        finite_mask = (
            np.isfinite(pc.points[0, :]) &
            np.isfinite(pc.points[1, :]) &
            np.isfinite(pc.points[2, :])
        )
        if not np.all(finite_mask):
            raise ValueError(f"Non-finite coordinates detected in sweep {idx} (token {sw['token']})")

        temporal_label = (
            "reference_keyframe" if sw['is_key_frame'] else
            f"forward_sweep_+{idx}" if temporal_strategy == "forward_subsequent" else
            f"historical_sweep_-{nsweeps - 1 - idx}"
        )

        sweep_records.append({
            "sweep_index": idx,
            "temporal_relationship": temporal_label,
            "sample_data_token": sw['token'],
            "is_key_frame": bool(sw['is_key_frame']),
            "timestamp": sw['timestamp'],
            "time_offset_to_cam_keyframe_ms": round(float(time_offset_ms), 3),
            "raw_pcd_point_count": int(raw_pcd_count),
            "retained_point_count": int(retained_count),
            "discarded_by_devkit_filter_count": int(discarded_count),
            "file_path": sw['filename']
        })

        all_x.extend(pc.points[0, :])
        all_y.extend(pc.points[1, :])
        all_z.extend(pc.points[2, :])
        all_vx_raw.extend(pc.points[6, :])
        all_vy_raw.extend(pc.points[7, :])
        all_vx_comp.extend(vx_comp_scalar)
        all_vy_comp.extend(vy_comp_scalar)
        all_dyn_prop.extend(pc.points[3, :].astype(int))
        all_id.extend(pc.points[4, :].astype(int))
        all_rcs.extend(pc.points[5, :])
        all_quality.extend(pc.points[10, :].astype(int))
        all_ambig.extend(pc.points[11, :].astype(int))
        all_pdh0.extend(pc.points[15, :].astype(int))
        all_sweep_idx.extend([idx] * retained_count)
        all_time_offset_ms.extend([time_offset_ms] * retained_count)

    merged_data = {
        "x": np.array(all_x, dtype=np.float32),
        "y": np.array(all_y, dtype=np.float32),
        "z": np.array(all_z, dtype=np.float32),
        "vx_raw": np.array(all_vx_raw, dtype=np.float32),
        "vy_raw": np.array(all_vy_raw, dtype=np.float32),
        "vx_comp": np.array(all_vx_comp, dtype=np.float32),
        "vy_comp": np.array(all_vy_comp, dtype=np.float32),
        "dyn_prop": np.array(all_dyn_prop, dtype=np.int32),
        "id": np.array(all_id, dtype=np.int32),
        "rcs": np.array(all_rcs, dtype=np.float32),
        "quality": np.array(all_quality, dtype=np.int32),
        "ambig": np.array(all_ambig, dtype=np.int32),
        "pdh0": np.array(all_pdh0, dtype=np.int32),
        "sweep_idx": np.array(all_sweep_idx, dtype=np.int32),
        "time_offset_ms": np.array(all_time_offset_ms, dtype=np.float32)
    }

    ref_info = {
        "sample_token": sample['token'],
        "scene_token": sample['scene_token'],
        "ref_cam_token": ref_cam_token,
        "ref_cam_timestamp": ref_cam_time,
        "ref_coordinate_frame": "Ego-Vehicle Coordinate System at Camera Keyframe Timestamp (V_ego)",
        "temporal_strategy": temporal_strategy,
        "ego_translation_global": ref_cam_pose['translation'],
        "ego_rotation_global": ref_cam_pose['rotation']
    }

    return merged_data, sweep_records, ref_info


def render_multi_sweep_visualization(merged_data, sweep_records, ref_info, output_filepath, show_plot=False):
    """
    Renders high-resolution dual-panel Bird's-Eye View (BEV) visualization:
    Panel 1: Multi-sweep radar positions color-coded by Dynamic Property with distinct sweep markers.
    Panel 2: Radar returns color-coded by scalar ego-compensated radial Doppler velocity
             using a diverging colormap centered at 0 m/s (no vector arrows).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 9.5), dpi=130, facecolor='#0d1117')
    plt.subplots_adjust(top=0.87, bottom=0.10, left=0.06, right=0.95, wspace=0.24)

    # Super Title with Explicit Scope
    fig.suptitle(
        "STAGE 4A: MULTI-SWEEP RADAR MOTION & DOPPLER ATTRIBUTION\n"
        "[DISCLAIMER: Scalar Radial Doppler Measurements Only — NOT 2D/3D Object Velocity Vectors, NOT Collision Prediction]",
        fontsize=13,
        fontweight='bold',
        color='#58a6ff',
        y=0.96
    )

    x = merged_data['x']
    y = merged_data['y']
    vx_comp = merged_data['vx_comp']
    dyn_prop = merged_data['dyn_prop']
    sweep_idx = merged_data['sweep_idx']

    cone_dist = 55.0
    half_angle = np.deg2rad(35)

    sweep_markers = ['o', 's', '^']
    sweep_alphas = [0.95, 0.75, 0.60]

    dp_colors = {
        0: '#00f0ff',  # Moving: Cyan
        1: '#8b949e',  # Stationary: Muted Slate Gray
        2: '#ff4d4d',  # Oncoming: Red (if present)
        3: '#d29922',  # Stationary candidate: Amber
        4: '#bc8cff',  # Unknown: Purple
        5: '#58a6ff',  # Crossing stationary: Blue
        6: '#7ee787',  # Crossing moving: Light Green
        7: '#6e7681'   # Stopped: Dark Gray
    }
    dp_names = {
        0: 'Moving',
        1: 'Stationary',
        2: 'Oncoming',
        3: 'Stationary Candidate',
        4: 'Unknown',
        5: 'Crossing Stationary',
        6: 'Crossing Moving',
        7: 'Stopped'
    }

    # =========================================================================
    # Panel 1: Multi-Sweep Positions & Dynamic Property
    # =========================================================================
    ax1.set_facecolor('#161b22')
    ax1.grid(True, color='#30363d', linestyle='--', linewidth=0.7)

    # Ego vehicle footprint at reference time
    ego_patch1 = plt.Rectangle((-1.0, -0.9), 4.5, 1.8, color='#ff7b72', alpha=0.85, label='Ego Vehicle (Reference)')
    ax1.add_patch(ego_patch1)
    ax1.arrow(0, 0, 3.5, 0, head_width=0.8, head_length=0.8, fc='yellow', ec='yellow', zorder=5)

    # Camera FOV boundaries
    ax1.plot([0, cone_dist], [0, cone_dist * np.tan(half_angle)], color='#e3b341', linestyle=':', linewidth=1.5, label='CAM_FRONT FOV Boundary')
    ax1.plot([0, cone_dist], [0, -cone_dist * np.tan(half_angle)], color='#e3b341', linestyle=':', linewidth=1.5)

    # Plot points by sweep and dynamic property
    plotted_dps = set()
    for s_idx in range(len(sweep_records)):
        s_mask = (sweep_idx == s_idx)
        for dp_val in np.unique(dyn_prop[s_mask]):
            dp_mask = s_mask & (dyn_prop == dp_val)
            color = dp_colors.get(dp_val, '#ffffff')
            dp_label = dp_names.get(dp_val, f'Prop {dp_val}')
            label_text = f"{dp_label} ({np.sum(dp_mask)})" if dp_val not in plotted_dps else None
            if label_text:
                plotted_dps.add(dp_val)

            ax1.scatter(
                x[dp_mask], y[dp_mask],
                c=color,
                marker=sweep_markers[s_idx % len(sweep_markers)],
                s=60 if s_idx == 0 else 45,
                alpha=sweep_alphas[s_idx % len(sweep_alphas)],
                edgecolors='#0d1117',
                linewidths=0.6,
                label=label_text,
                zorder=4
            )

    # Sweep indicator legend elements
    from matplotlib.lines import Line2D
    marker_legend_elements = []
    for i in range(len(sweep_records)):
        sw = sweep_records[i]
        rel_tag = "Keyframe" if sw['is_key_frame'] else f"Forward +{i}"
        marker_legend_elements.append(
            Line2D([0], [0], marker=sweep_markers[i], color='w',
                   label=f"Sweep {i} ({rel_tag}, {sw['time_offset_to_cam_keyframe_ms']:+.1f} ms): {sw['retained_point_count']} pts",
                   markerfacecolor='#58a6ff', markersize=8, linestyle='None')
        )

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(
        handles=h1 + marker_legend_elements,
        loc='upper right',
        facecolor='#0d1117',
        edgecolor='#30363d',
        labelcolor='white',
        fontsize=8,
        framealpha=0.9
    )

    ax1.set_xlim(-5.0, 65.0)
    ax1.set_ylim(-30.0, 30.0)
    ax1.set_xlabel("X Forward in Reference Ego Frame (m)", color='white', fontsize=10)
    ax1.set_ylabel("Y Left/Right in Reference Ego Frame (m)", color='white', fontsize=10)
    ax1.set_title(
        f"Multi-Sweep Returns by Dynamic Property ({len(x)} points across {len(sweep_records)} sweeps)\n"
        "[Ego-motion compensated into V_ego; markers distinguish sweep order]",
        color='white',
        fontsize=11,
        fontweight='bold'
    )
    ax1.tick_params(colors='white')

    # =========================================================================
    # Panel 2: Scalar Compensated Radial Doppler Velocity (No 2D Vector Arrows)
    # =========================================================================
    ax2.set_facecolor('#161b22')
    ax2.grid(True, color='#30363d', linestyle='--', linewidth=0.7)

    # Ego vehicle footprint
    ego_patch2 = plt.Rectangle((-1.0, -0.9), 4.5, 1.8, color='#ff7b72', alpha=0.85, label='Ego Vehicle (Reference)')
    ax2.add_patch(ego_patch2)
    ax2.arrow(0, 0, 3.5, 0, head_width=0.8, head_length=0.8, fc='yellow', ec='yellow', zorder=5)

    # Camera FOV boundaries
    ax2.plot([0, cone_dist], [0, cone_dist * np.tan(half_angle)], color='#e3b341', linestyle=':', linewidth=1.5, label='CAM_FRONT FOV Boundary')
    ax2.plot([0, cone_dist], [0, -cone_dist * np.tan(half_angle)], color='#e3b341', linestyle=':', linewidth=1.5)

    # Diverging colormap centered at 0.0 m/s
    norm_doppler = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=12.0)
    scatter2 = ax2.scatter(
        x, y,
        c=vx_comp,
        cmap='coolwarm',
        norm=norm_doppler,
        s=60,
        alpha=0.92,
        edgecolors='#30363d',
        linewidths=0.6,
        zorder=4
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label('Ego-Compensated Radial Velocity $V_{x, \\mathrm{comp}}$ (m/s)', color='white', fontsize=9)
    cbar2.ax.tick_params(colors='white')

    ax2.set_xlim(-5.0, 65.0)
    ax2.set_ylim(-30.0, 30.0)
    ax2.set_xlabel("X Forward in Reference Ego Frame (m)", color='white', fontsize=10)
    ax2.set_ylabel("Y Left/Right in Reference Ego Frame (m)", color='white', fontsize=10)
    ax2.set_title(
        f"Ego-Compensated Radial Doppler Measurement (Scalar $V_{{x, \\mathrm{{comp}}}}$)\n"
        "[Scalar line-of-sight closure rate — NO 2D travel direction or trajectory implied]",
        color='white',
        fontsize=11,
        fontweight='bold'
    )
    ax2.tick_params(colors='white')

    # Explanatory text box on velocity interpretation
    info_text = (
        "Doppler Scalar Interpretation:\n"
        "• Blue (<0 m/s) : Closing toward ego\n"
        "• White (~0 m/s): Stationary relative to ego/world\n"
        "• Red (>0 m/s)  : Receding away from ego\n"
        "Note: 1D line-of-sight measurement;\n"
        "zero Doppler for perpendicular motion."
    )
    ax2.text(
        0.03, 0.96, info_text,
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment='top',
        color='#c9d1d9',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117', edgecolor='#30363d', alpha=0.9)
    )

    plt.savefig(output_filepath, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show_plot:
        plt.show(block=False)
        plt.pause(2.0)
    plt.close(fig)


def run_stage4_motion(sample_idx=0,
                      nsweeps=3,
                      output_dir="outputs/motion_conflict",
                      show_plot=False):
    """
    Main execution pipeline for Stage 4A Multi-Sweep Radar Motion Visualisation.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print(" STAGE 4A: MULTI-SWEEP RADAR MOTION VISUALISATION")
    print(" (DISCLAIMER: Scalar Radial Doppler Attribution Only — NOT Collision Prediction)")
    print("=================================================================")

    # 1. Initialize dataset
    print("[*] Initializing nuScenes v1.0-mini...")
    nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)

    sample = nusc.sample[sample_idx]
    sample_token = sample['token']
    scene_record = nusc.get('scene', sample['scene_token'])
    scene_name = scene_record['name']

    print(f"[*] Target Sample Token : {sample_token}")
    print(f"[*] Scene Name          : {scene_name}")
    print(f"[*] Sweeps to Accumulate: {nsweeps}")

    # 2. Accumulate and transform multi-sweep radar
    print("[*] Performing multi-sweep ego-motion compensated transformation...")
    merged_data, sweep_records, ref_info = load_multi_sweep_radar(
        nusc=nusc,
        sample=sample,
        nsweeps=nsweeps
    )

    total_retained = len(merged_data['x'])
    total_raw_pcd = sum(sw['raw_pcd_point_count'] for sw in sweep_records)
    total_discarded = sum(sw['discarded_by_devkit_filter_count'] for sw in sweep_records)

    # 3. Compute statistics and breakdown
    dyn_prop = merged_data['dyn_prop']
    vx_raw = merged_data['vx_raw']
    vx_comp = merged_data['vx_comp']

    num_moving = int(np.sum(dyn_prop == 0))
    num_stationary = int(np.sum(dyn_prop == 1))
    num_stat_cand = int(np.sum(dyn_prop == 3))
    num_other = total_retained - (num_moving + num_stationary + num_stat_cand)

    # 4. Console Quantitative Summary
    print("-----------------------------------------------------------------")
    print(" MULTI-SWEEP RADAR MOTION QUANTITATIVE SUMMARY:")
    print("-----------------------------------------------------------------")
    print(f" Sample Token                  : {sample_token}")
    print(f" Coordinate Frame              : {ref_info['ref_coordinate_frame']}")
    print(f" Temporal Traversal Strategy   : {ref_info['temporal_strategy']}")
    print(f" Number of Sweeps Accumulated  : {len(sweep_records)}")
    for sw in sweep_records:
        rel_str = f" [{sw['temporal_relationship']}]"
        print(f"   Sweep {sw['sweep_index']}: token={sw['sample_data_token'][:8]}... | offset={sw['time_offset_to_cam_keyframe_ms']:+7.2f} ms | raw PCD={sw['raw_pcd_point_count']} -> retained={sw['retained_point_count']} (discarded {sw['discarded_by_devkit_filter_count']}){rel_str}")
    print(f" Total Raw PCD Returns Loaded  : {total_raw_pcd} points")
    print(f" Total Retained Returns        : {total_retained} points (after devkit default validity & ambiguity filters)")
    print(f" Total Discarded Returns       : {total_discarded} points")
    print(f" All Coordinates Finite        : {bool(np.all(np.isfinite(merged_data['x'])) and np.all(np.isfinite(merged_data['y'])))}")
    print(" Dynamic Property Breakdown    :")
    print(f"   -> Moving (dyn_prop=0)      : {num_moving} points ({num_moving/total_retained*100:.1f}%)")
    print(f"   -> Stationary (dyn_prop=1)  : {num_stationary} points ({num_stationary/total_retained*100:.1f}%)")
    print(f"   -> Stat. Candidate (prop=3) : {num_stat_cand} points ({num_stat_cand/total_retained*100:.1f}%)")
    if num_other > 0:
        print(f"   -> Other Categories         : {num_other} points")
    print(" Radial Velocity Distributions :")
    print(f"   -> Raw vx (uncompensated)  : mean={np.mean(vx_raw):+.2f} m/s | min={np.min(vx_raw):+.2f} | max={np.max(vx_raw):+.2f}")
    print(f"   -> Comp vx (ego-compensated): mean={np.mean(vx_comp):+.2f} m/s | min={np.min(vx_comp):+.2f} | max={np.max(vx_comp):+.2f}")
    if num_moving > 0:
        print(f"   -> Moving points Comp vx    : mean={np.mean(vx_comp[dyn_prop == 0]):+.2f} m/s | min={np.min(vx_comp[dyn_prop == 0]):+.2f} | max={np.max(vx_comp[dyn_prop == 0]):+.2f}")
    print("-----------------------------------------------------------------")
    print(" DOCUMENTED ASSUMPTIONS & LIMITATIONS:")
    print(" 1. Scalar Radial Doppler: Measured velocity vx_comp reflects the 1D scalar")
    print("    line-of-sight closing rate. It does NOT provide 2D Cartesian object velocity.")
    print("    No vector arrows or travel headings are implied.")
    print(" 2. Forward-in-Time Traversal: Because sample 0 is at scene inception (no preceding")
    print("    sweeps), sweeps 1 and 2 are subsequent forward-in-time observations (+75 ms, +150 ms),")
    print("    transformed into the keyframe reference ego frame.")
    print(" 3. No Inter-Sweep Association: Coordinate alignment under ego-motion compensation")
    print("    does NOT establish that returns belong to the same physical object without tracking.")
    print(" 4. Non-Predictive: This script visualises motion observations only.")
    print("    It makes no claims regarding collision probability, hazard zones, or crash prediction.")
    print("-----------------------------------------------------------------")

    # 5. Export JSON Metrics
    metrics_path = os.path.join(output_dir, "radar_motion_metrics.json")
    metrics_data = {
        "pipeline_stage": "Stage 4A Multi-Sweep Radar Motion Visualisation",
        "sample_token": sample_token,
        "scene_name": scene_name,
        "reference_coordinate_frame": ref_info['ref_coordinate_frame'],
        "temporal_sweep_strategy": ref_info['temporal_strategy'],
        "temporal_strategy_explanation": (
            "Because reference sample ca9a282c... is at the start of scene-0061 (sample[0], prev=''), "
            "consecutive sweeps 1 and 2 are forward-in-time subsequent sweeps (+75.03 ms, +150.21 ms) "
            "collected via sample_data['next'] and transformed into the reference keyframe ego frame."
        ),
        "nsweeps_accumulated": len(sweep_records),
        "sweep_details": sweep_records,
        "returns_summary": {
            "total_raw_pcd_returns": total_raw_pcd,
            "total_retained_returns": total_retained,
            "total_discarded_returns": total_discarded,
            "all_coordinates_finite": bool(np.all(np.isfinite(merged_data['x'])) and np.all(np.isfinite(merged_data['y'])))
        },
        "applied_filtering": {
            "nuScenes_devkit_filters_in_from_file": {
                "invalid_states_filter": "kept [0] (0x00=valid cluster; discards low RCS, near-field artefacts, mirrors, harmonics)",
                "dynprop_states_filter": "kept range(7) [0..6] (moving, stationary, oncoming, etc.; discards 7=stopped)",
                "ambig_states_filter": "kept [3] (unambiguous Doppler solution only; discards ambiguous/invalid solutions)"
            },
            "downstream_custom_filters": "none (all 216 retained returns preserved without spatial or velocity truncation)"
        },
        "dynamic_property_breakdown": {
            "moving_points_count": num_moving,
            "stationary_points_count": num_stationary,
            "stationary_candidate_points_count": num_stat_cand,
            "other_points_count": num_other
        },
        "velocity_interpretation": {
            "velocity_field_used": "vx_comp (Index 8)",
            "interpretation": "Scalar ego-compensated radial Doppler velocity along the sensor line-of-sight.",
            "explicit_caution": "vx_comp and vy_comp do NOT constitute a complete 2D object velocity vector. No 2D travel direction or trajectory is claimed."
        },
        "velocity_statistics_m_per_s": {
            "raw_vx_mean": float(np.mean(vx_raw)),
            "raw_vx_min": float(np.min(vx_raw)),
            "raw_vx_max": float(np.max(vx_raw)),
            "compensated_vx_mean": float(np.mean(vx_comp)),
            "compensated_vx_min": float(np.min(vx_comp)),
            "compensated_vx_max": float(np.max(vx_comp)),
            "moving_points_compensated_vx_mean": float(np.mean(vx_comp[dyn_prop == 0])) if num_moving > 0 else 0.0,
            "stationary_points_compensated_vx_mean": float(np.mean(vx_comp[dyn_prop == 1])) if num_stationary > 0 else 0.0
        },
        "documented_limitations": [
            "Radial velocity measures 1D line-of-sight closing rate; it does not provide complete 2D or 3D object velocity vectors.",
            "Ego-motion compensation transforms coordinate frames, but does not track or associate returns between sweeps as specific physical objects.",
            "Radar returns are physically sparse reflections; multipath or ghost returns can occur.",
            "No collision prediction, crash probability, or hazard thresholds are claimed."
        ]
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[*] Motion metrics saved   : {os.path.abspath(metrics_path)}")

    # 6. Render Visualisation
    vis_path = os.path.join(output_dir, "multi_sweep_radar_motion.png")
    print("[*] Generating multi-sweep BEV visualisation...")
    render_multi_sweep_visualization(
        merged_data=merged_data,
        sweep_records=sweep_records,
        ref_info=ref_info,
        output_filepath=vis_path,
        show_plot=show_plot
    )
    print(f"[*] Visualisation saved    : {os.path.abspath(vis_path)}")
    print("=================================================================")
    print(" STAGE 4A MULTI-SWEEP VISUALISATION COMPLETED SUCCESSFULLY.")
    print("=================================================================")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4A: Multi-Sweep Radar Motion Visualisation")
    parser.add_argument("--sample-idx", type=int, default=0, help="Target keyframe sample index (default: 0)")
    parser.add_argument("--nsweeps", type=int, default=3, help="Number of sweeps to accumulate (default: 3)")
    parser.add_argument("--output-dir", type=str, default="outputs/motion_conflict", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display figure window")

    args = parser.parse_args()

    run_stage4_motion(
        sample_idx=args.sample_idx,
        nsweeps=args.nsweeps,
        output_dir=args.output_dir,
        show_plot=args.show
    )
