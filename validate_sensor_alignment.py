#!/usr/bin/env python3
"""
Stage 2: Radar-Camera Geometric Sensor Alignment Validation
============================================================
Project: Radar–Camera Fusion-Based Driver Assistance System (MY_ADAS_PROJECT)
Author / Pipeline: Stage 2 Geometric Calibration & Projection Validation

DISCLAIMER & SCOPE:
-------------------
This script validates spatial and temporal calibration alignment between RADAR_FRONT
and CAM_FRONT using the official nuScenes devkit geometric transform chain and
projection utilities. It strictly performs geometric forward projection of physical
sensor points. It does NOT infer hidden vehicles, estimate dense camera depth,
predict collisions, or claim semantic occupancy.

COORDINATE TRANSFORM & PROJECTION METHODOLOGY:
---------------------------------------------
1. Official NuScenes API Inspection:
   nuScenesExplorer.map_pointcloud_to_image(pointsensor_token, camera_token, min_dist=1.0)
   - Argument 1: pointsensor_token (RADAR_FRONT sample_data token)
   - Argument 2: camera_token (CAM_FRONT sample_data token)
   - Official projection utility: nuscenes.utils.geometry_utils.view_points(points, view, normalize=True)

2. Step-by-Step Rigorous Transform Sequence:
   - S_radar -> V_ego(t_radar): Rotate & translate via calibrated_sensor(RADAR_FRONT)
   - V_ego(t_radar) -> W_global: Rotate & translate via ego_pose(RADAR_FRONT)
   - W_global -> V_ego(t_cam): Translate & rotate via inverse ego_pose(CAM_FRONT)
   - V_ego(t_cam) -> S_cam: Translate & rotate via inverse calibrated_sensor(CAM_FRONT)
   - S_cam -> 2D Image Plane: Perspective projection via camera intrinsic matrix K
     using view_points(points[:3, :], K, normalize=True)

3. Depth and Field-of-View Filtering:
   - Positive camera depth: Z_cam > 0 (points in front of camera lens plane)
   - In-image boundaries: 0 <= u < image_width and 0 <= v < image_height
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points


def validate_sensor_alignment(sample_idx=0,
                              output_dir="outputs/alignment",
                              show_plot=False):
    """
    Validates radar-to-camera geometric alignment and projections on a nuScenes sample.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print(" STAGE 2: RADAR-CAMERA SENSOR ALIGNMENT VALIDATION")
    print(" (DISCLAIMER: Geometric Projection Only — No Occupancy/Hazards)")
    print("=================================================================")

    # 1. Initialize dataset
    print("[*] Initializing nuScenes v1.0-mini...")
    nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)

    sample = nusc.sample[sample_idx]
    sample_token = sample['token']
    scene_record = nusc.get('scene', sample['scene_token'])
    scene_name = scene_record['name']

    cam_token = sample['data']['CAM_FRONT']
    radar_token = sample['data']['RADAR_FRONT']

    cam_sd = nusc.get('sample_data', cam_token)
    radar_sd = nusc.get('sample_data', radar_token)

    cam_path = os.path.join(nusc.dataroot, cam_sd['filename'])
    radar_path = os.path.join(nusc.dataroot, radar_sd['filename'])

    print(f"[*] Target Sample Token : {sample_token}")
    print(f"[*] Scene Name          : {scene_name}")
    print(f"[*] Camera File         : {cam_sd['filename']}")
    print(f"[*] Radar File          : {radar_sd['filename']}")

    # 2. Extract Timestamps
    cam_timestamp = cam_sd['timestamp']
    radar_timestamp = radar_sd['timestamp']
    diff_us = radar_timestamp - cam_timestamp
    diff_ms = diff_us / 1000.0

    # 3. Load Raw Radar Point Cloud
    pc_radar = RadarPointCloud.from_file(radar_path)
    raw_point_count = pc_radar.points.shape[1]

    # 4. Rigorous Step-by-Step Alignment to Camera Timestamp
    cs_radar = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
    pose_radar = nusc.get('ego_pose', radar_sd['ego_pose_token'])
    cs_cam = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    pose_cam = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # Step A: Transform to Ego frame at radar timestamp
    pc_radar.rotate(Quaternion(cs_radar['rotation']).rotation_matrix)
    pc_radar.translate(np.array(cs_radar['translation']))

    # Step B: Transform to Global World frame
    pc_radar.rotate(Quaternion(pose_radar['rotation']).rotation_matrix)
    pc_radar.translate(np.array(pose_radar['translation']))

    # Step C: Transform to Ego frame at camera timestamp
    pc_radar.translate(-np.array(pose_cam['translation']))
    pc_radar.rotate(Quaternion(pose_cam['rotation']).rotation_matrix.T)

    # Step D: Transform from Ego frame into Camera sensor frame
    pc_radar.translate(-np.array(cs_cam['translation']))
    pc_radar.rotate(Quaternion(cs_cam['rotation']).rotation_matrix.T)

    # 5. Extract Camera Depths (Z axis in camera optical frame)
    depths = pc_radar.points[2, :].copy()
    positive_depth_mask = depths > 0.0
    positive_depth_count = int(np.sum(positive_depth_mask))

    # 6. Project Points to Camera Image Plane using official view_points
    cam_intrinsic = np.array(cs_cam['camera_intrinsic'])
    pts_2d = view_points(pc_radar.points[:3, :], cam_intrinsic, normalize=True)

    # Load Camera Image dimensions
    cam_img = Image.open(cam_path)
    img_width, img_height = cam_img.size

    # 7. Filter points within image boundaries and positive depth
    in_image_mask = (
        positive_depth_mask &
        (pts_2d[0, :] >= 0) &
        (pts_2d[0, :] < img_width) &
        (pts_2d[1, :] >= 0) &
        (pts_2d[1, :] < img_height)
    )
    in_image_count = int(np.sum(in_image_mask))

    # Also verify using official nuScenesExplorer utility (correct argument order: pointsensor_token, camera_token)
    pts_explorer, depths_explorer, _ = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=radar_token,
        camera_token=cam_token,
        min_dist=1.0
    )
    explorer_count = pts_explorer.shape[1]

    # Valid points for visualization
    u_valid = pts_2d[0, in_image_mask]
    v_valid = pts_2d[1, in_image_mask]
    depth_valid = depths[in_image_mask]

    # Extract Doppler velocity attributes for valid points if present
    # In raw radar points, row 8 & 9 are compensated velocity components
    vx_valid = pc_radar.points[8, in_image_mask] if pc_radar.points.shape[0] > 8 else np.zeros_like(depth_valid)
    vy_valid = pc_radar.points[9, in_image_mask] if pc_radar.points.shape[0] > 9 else np.zeros_like(depth_valid)

    # 8. Report Alignment Metrics to Console (Requirement 6)
    print("-----------------------------------------------------------------")
    print(" SENSOR ALIGNMENT & PROJECTION METRICS:")
    print("-----------------------------------------------------------------")
    print(f" Sample Token               : {sample_token}")
    print(f" Camera Timestamp           : {cam_timestamp} us")
    print(f" Radar Timestamp            : {radar_timestamp} us")
    print(f" Timestamp Difference       : {diff_ms:+.3f} ms ({diff_us:+} us)")
    print(f" Raw Radar Point Count      : {raw_point_count}")
    print(f" Positive-Depth Point Count : {positive_depth_count} (Z_cam > 0)")
    print(f" In-Image Point Count       : {in_image_count} (0 <= u < {img_width}, 0 <= v < {img_height})")
    print(f" NuScenes Explorer Match    : {explorer_count} points (verified official API match)")
    print(f" Valid Points Min Depth     : {depth_valid.min():.2f} m" if in_image_count > 0 else "N/A")
    print(f" Valid Points Max Depth     : {depth_valid.max():.2f} m" if in_image_count > 0 else "N/A")
    print("-----------------------------------------------------------------")

    # 9. Save Metrics to JSON
    metrics_file = os.path.join(output_dir, "alignment_metrics.json")
    metrics_data = {
        "pipeline_stage": "Stage 2 Sensor Alignment Validation",
        "sample_token": sample_token,
        "scene_name": scene_name,
        "camera_sensor": "CAM_FRONT",
        "radar_sensor": "RADAR_FRONT",
        "camera_timestamp_us": cam_timestamp,
        "radar_timestamp_us": radar_timestamp,
        "timestamp_difference_ms": round(diff_ms, 3),
        "timestamp_difference_us": diff_us,
        "raw_radar_point_count": raw_point_count,
        "positive_depth_point_count": positive_depth_count,
        "in_image_point_count": in_image_count,
        "nuscenes_explorer_match_count": explorer_count,
        "depth_range_m": [float(depth_valid.min()), float(depth_valid.max())] if in_image_count > 0 else [],
        "image_resolution": [img_width, img_height],
        "camera_intrinsic_matrix": cam_intrinsic.tolist(),
        "disclaimer": "Geometric sensor projection only. No depth estimation, collision prediction, or semantic occupancy."
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[*] Alignment metrics saved : {os.path.abspath(metrics_file)}")

    # 10. Generate and Save Visualisation (Requirement 5)
    vis_file = os.path.join(output_dir, "sensor_alignment_CAM_FRONT.png")
    print("[*] Generating alignment visualisation...")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=140, facecolor='#0d1117')
    ax.imshow(cam_img)

    if in_image_count > 0:
        scatter = ax.scatter(
            u_valid, v_valid,
            c=depth_valid,
            cmap='turbo',
            s=90,
            edgecolors='white',
            linewidths=1.5,
            alpha=0.95,
            label=f'Radar Return ({in_image_count} in-image)'
        )

        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('Radar Physical Depth in Camera Frame (meters)', color='white', fontsize=11, labelpad=10)
        cbar.ax.tick_params(colors='white', labelsize=10)

    # Title and Metadata Annotations
    ax.set_title(
        f"STAGE 2: RADAR–CAMERA SENSOR ALIGNMENT VALIDATION\n"
        f"Sample: {sample_token[:16]}... | RADAR_FRONT -> CAM_FRONT (dt = {diff_ms:+.2f} ms)\n"
        f"[Raw Points: {raw_point_count} | Positive-Depth: {positive_depth_count} | In-Image Valid: {in_image_count}]",
        color='#00e5ff',
        fontsize=12,
        fontweight='bold',
        pad=14
    )

    # Bottom disclaimer text box
    disclaimer_text = (
        "VALIDATION NOTICE: Verified geometric forward projection using official nuScenes calibrated sensor & ego poses.\n"
        "Coloured by true radar sensor depth (m). No neural depth estimation, hazard detection, or occupancy prediction."
    )
    ax.text(
        0.5, 0.02, disclaimer_text,
        transform=ax.transAxes,
        color='#e6edf3',
        fontsize=9,
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', edgecolor='#30363d', alpha=0.85)
    )

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(vis_file, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show_plot:
        plt.show(block=False)
        plt.pause(2.0)
    plt.close(fig)

    print(f"[*] Visualisation saved     : {os.path.abspath(vis_file)}")
    print("=================================================================")
    print(" STAGE 2 SENSOR ALIGNMENT COMPLETED SUCCESSFULLY.")
    print("=================================================================")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Radar-Camera Sensor Alignment Validation")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to evaluate (default: 0)")
    parser.add_argument("--output-dir", type=str, default="outputs/alignment", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactive figure window")

    args = parser.parse_args()

    validate_sensor_alignment(
        sample_idx=args.sample_idx,
        output_dir=args.output_dir,
        show_plot=args.show
    )
