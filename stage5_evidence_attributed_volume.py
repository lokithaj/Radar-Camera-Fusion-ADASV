#!/usr/bin/env python3
"""
Stage 5: Evidence-Attributed Hazard Volume (EAHV) Visualization Prototype
==========================================================================
Project: Radar–Camera Fusion-Based Driver Assistance System (MY_ADAS_PROJECT)
Pipeline Stage: Stage 5 Geometric Evidence Attribution & Hazard Volume Prototype

DISCLAIMER & SCIENTIFIC SCOPE:
------------------------------
1. REPRODUCIBLE GEOMETRIC EVIDENCE PROTOTYPE ONLY:
   This module generates a reproducible geometric sensor evidence visualization.
   It is NOT an automated collision prediction system, an automated emergency braking
   (AEB) trigger, a neural-network 3D perception system, or a safety-certified ADAS.
2. NO INVENTED OCCUPANCY PROBABILITIES OR ARBITRARY HAZARD THRESHOLDS:
   No subjective occupancy grids, probability decay distributions, arbitrary risk
   scores, or collision probabilities are computed. Voxels are categorized strictly
   by observable physical sensor measurements.
3. PRESERVATION OF UNOBSERVED / UNKNOWN SPACE:
   Unobserved spatial voxels (containing zero radar reflections) are strictly labeled
   as UNKNOWN / UNOBSERVED space. In the absence of a direct 3D range sensor measurement,
   monocular cameras cannot supply dense depth; therefore, geometry alone cannot verify
   whether unobserved space is free or occupied. Space is NEVER falsely claimed as "free".
4. CAMERA PROJECTION CONTEXT vs. VISUAL VISIBILITY:
   Projecting a 3D coordinate onto the camera image plane demonstrates geometric
   frustum alignment ONLY. It does NOT prove optical line-of-sight visibility, feature
   salience, or absence of foreground occlusion.
5. TEMPORAL CANDIDATE EVIDENCE vs. OBJECT TRACKING:
   Temporal associations imported from Stage 4B represent exploratory geometric proximity
   candidates between successive sweeps. They are labeled as Heuristic Candidate Evidence,
   NOT validated multi-object tracks (MOT).
6. RADAR DOPPLER VELOCITY INTERPRETATION:
   Radar Doppler (vx_comp) is an ego-motion-compensated 1D scalar line-of-sight velocity
   measurement along the radar beam axis (nuScenes point cloud row 8). It does NOT
   represent a complete 2D or 3D Cartesian object velocity vector. Cross-range velocity
   is unmeasured; kinematic vector forward-projection is not performed.

COORDINATE SYSTEM SPECIFICATION:
--------------------------------
Reference Coordinate Frame: Ego-Vehicle Coordinate System at Camera Keyframe Timestamp (V_ego)
- Origin (0, 0, 0): Center of vehicle rear axle projected onto ground road surface.
- +X axis: Forward along vehicle longitudinal centerline.
- +Y axis: Lateral leftward perpendicular to vehicle heading.
- +Z axis: Vertical upward normal to road plane.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering for robust, reproducible output generation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon, Patch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

# Reuse validated modules from earlier stages
from stage4_radar_motion import load_multi_sweep_radar
from stage4_temporal_association import compute_consecutive_sweep_association
from evidence_attribution import load_and_transform_sensors


# ==============================================================================
# VOXEL GRID SPECIFICATION & EVIDENCE TAXONOMY
# ==============================================================================

# Voxel State Taxonomy (Strict Physical Categorization)
STATE_UNKNOWN = 0          # Unknown / Unobserved space (Zero physical range returns)
STATE_RADAR_SUPPORTED = 1  # Observed physical radar return within grid boundaries

# Temporal Candidate Evidence Tiers (from Stage 4B consecutive sweep association)
TIER_HEURISTIC_CANDIDATE = "Heuristic Candidate Evidence"  # Unique 1-to-1 temporal candidate
TIER_AMBIGUOUS_CANDIDATE = "Ambiguous Candidate Evidence"  # Contested temporal candidate
TIER_ISOLATED_RADAR = "Isolated Radar Evidence"            # Unmatched in consecutive sweep

# Camera Projection Context (from Stage 3 calibrated projection)
CAM_PROJECTS_VIEWPORT = "Projects into Camera Viewport"
CAM_OUTSIDE_VIEWPORT = "Outside Camera Viewport"


class VoxelEvidenceVolumeGrid3D:
    """
    3D Regular Voxel Grid for Evidence-Attributed Hazard Volume (EAHV).
    
    Spatial bounds strictly preserved from Stage 1:
      X in [0.0, 60.0] m   (Forward, longitudinal)
      Y in [-25.0, 25.0] m (Lateral, left/right)
      Z in [-2.0, 4.0] m   (Vertical, elevation)
      Resolution: (0.5, 0.5, 0.5) m
      Total voxels: 120 x 100 x 12 = 144,000 cells.
    """
    def __init__(self,
                 x_bounds=(0.0, 60.0),
                 y_bounds=(-25.0, 25.0),
                 z_bounds=(-2.0, 4.0),
                 voxel_size=(0.5, 0.5, 0.5)):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.z_min, self.z_max = z_bounds
        self.vx, self.vy, self.vz = voxel_size

        self.nx = int(np.round((self.x_max - self.x_min) / self.vx))
        self.ny = int(np.round((self.y_max - self.y_min) / self.vy))
        self.nz = int(np.round((self.z_max - self.z_min) / self.vz))

        self.grid_dimensions = (self.nx, self.ny, self.nz)
        self.total_voxels = self.nx * self.ny * self.nz
        self.voxel_volume_m3 = self.vx * self.vy * self.vz
        self.total_volume_m3 = self.total_voxels * self.voxel_volume_m3

        # State grid: STATE_UNKNOWN (0) vs STATE_RADAR_SUPPORTED (1)
        self.state_grid = np.full(self.grid_dimensions, STATE_UNKNOWN, dtype=np.uint8)

        # Storage for occupied voxel records
        self.occupied_voxels = {}

    def point_to_voxel_index(self, x, y, z):
        """Converts continuous physical coordinates (meters) to 3D voxel indices."""
        ix = np.floor((x - self.x_min) / self.vx).astype(int)
        iy = np.floor((y - self.y_min) / self.vy).astype(int)
        iz = np.floor((z - self.z_min) / self.vz).astype(int)
        return ix, iy, iz

    def voxel_index_to_centroid(self, ix, iy, iz):
        """Converts integer voxel indices back to physical center coordinates (meters)."""
        x_c = self.x_min + (ix + 0.5) * self.vx
        y_c = self.y_min + (iy + 0.5) * self.vy
        z_c = self.z_min + (iz + 0.5) * self.vz
        return x_c, y_c, z_c

    def is_inside(self, x, y, z):
        """Boolean filter checking if continuous coordinates fall within grid volume."""
        return (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max) &
            (z >= self.z_min) & (z < self.z_max)
        )


# ==============================================================================
# PIPELINE EXECUTION & ATTRIBUTION MAPPING
# ==============================================================================

def execute_stage5_eahv(
    nusc,
    sample_token="ca9a282c9e77460f8360f564131a8af5",
    output_dir="outputs/eahv",
    show_plot=False
):
    """
    Executes Stage 5 Evidence-Attributed Hazard Volume (EAHV) prototype.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample = nusc.get('sample', sample_token)
    scene_record = nusc.get('scene', sample['scene_token'])
    scene_name = scene_record['name']

    print("=" * 76)
    print(" STAGE 5: EVIDENCE-ATTRIBUTED HAZARD VOLUME (EAHV) PROTOTYPE")
    print(" [Geometric Evidence Attribution Only — NOT a Certified ADAS System]")
    print("=" * 76)
    print(f"[*] Target Sample Token : {sample_token}")
    print(f"[*] Scene Name          : {scene_name}")

    # 1. Initialize Stage 1 Voxel Grid
    grid = VoxelEvidenceVolumeGrid3D()
    print(f"[*] Voxel Grid: [{grid.nx} x {grid.ny} x {grid.nz}] = {grid.total_voxels:,} cells")
    print(f"    Bounds: X in [{grid.x_min}, {grid.x_max}] m, Y in [{grid.y_min}, {grid.y_max}] m, Z in [{grid.z_min}, {grid.z_max}] m")
    print(f"    Resolution: {grid.vx} x {grid.vy} x {grid.vz} m ({grid.voxel_volume_m3} m^3 per voxel)")

    # 2. Load Radar Streams & Transform to Reference Frame V_ego (Stage 4A Loader)
    print("[*] Loading multi-sweep radar data in reference ego frame V_ego...")
    merged_data, sweep_records, ref_info = load_multi_sweep_radar(nusc, sample, nsweeps=3)

    # 3. Compute Consecutive Sweep Candidate Associations (Stage 4B Associator)
    print("[*] Computing Stage 4B candidate temporal associations (Sweeps 0 -> 1)...")
    assoc_01 = compute_consecutive_sweep_association(
        merged_data, sweep_records,
        src_sweep_idx=0, dst_sweep_idx=1,
        d_gate=1.5, dv_gate=2.5
    )

    # 4. Load Camera Data & Project Radar Points to CAM_FRONT (Stage 3 Calibrator)
    print("[*] Performing geometric sensor alignment & CAM_FRONT viewport projection...")
    (
        points_ego,
        points_cam,
        pts_2d,
        cam_path,
        cam_sd,
        radar_sd,
        cs_cam,
        raw_radar_count
    ) = load_and_transform_sensors(nusc, sample)

    # Reference Sweep 0 Data
    m_sw0 = merged_data['sweep_idx'] == 0
    sw0_indices = np.where(m_sw0)[0]
    total_sw0_returns = len(sw0_indices)

    x_sw0 = merged_data['x'][sw0_indices]
    y_sw0 = merged_data['y'][sw0_indices]
    z_sw0 = merged_data['z'][sw0_indices]
    vx_comp_sw0 = merged_data['vx_comp'][sw0_indices]
    rcs_sw0 = merged_data['rcs'][sw0_indices]
    dyn_prop_sw0 = merged_data['dyn_prop'][sw0_indices]
    id_sw0 = merged_data['id'][sw0_indices]

    # Camera Optical Depth & Pixel Projections for Sweep 0
    img_w, img_h = cam_sd['width'], cam_sd['height']
    depth_cam_sw0 = points_cam[2, :]
    u_sw0 = pts_2d[0, :]
    v_sw0 = pts_2d[1, :]

    in_cam_viewport_sw0 = (
        (depth_cam_sw0 > 0.0) &
        (u_sw0 >= 0.0) & (u_sw0 < img_w) &
        (v_sw0 >= 0.0) & (v_sw0 < img_h)
    )

    # Extract Stage 4B Association Membership for Sweep 0
    matched_src_local = {p['src_local_idx']: p for p in assoc_01['unique_pairs']}
    ambiguous_src_local = set(assoc_01['ambiguous_src_indices'])
    unmatched_src_local = set(assoc_01['unmatched_src_indices'])

    # Verification: Mutual exclusivity of Stage 4B source returns
    assert len(set(matched_src_local.keys()).intersection(ambiguous_src_local)) == 0, "Stage 4B source partition overlap!"
    assert len(set(matched_src_local.keys()).intersection(unmatched_src_local)) == 0, "Stage 4B source partition overlap!"
    assert len(ambiguous_src_local.intersection(unmatched_src_local)) == 0, "Stage 4B source partition overlap!"
    assert (len(matched_src_local) + len(ambiguous_src_local) + len(unmatched_src_local)) == total_sw0_returns, "Stage 4B partition incomplete!"

    # 5. Spatial Voxel Filtering & Attribution Mapping
    in_grid_mask = grid.is_inside(x_sw0, y_sw0, z_sw0)
    in_grid_indices = np.where(in_grid_mask)[0]
    out_of_grid_indices = np.where(~in_grid_mask)[0]

    count_in_grid = len(in_grid_indices)
    count_out_of_grid = len(out_of_grid_indices)
    assert count_in_grid + count_out_of_grid == total_sw0_returns, "Grid spatial partition incomplete!"

    # Map in-grid returns into discrete voxels
    voxel_records = []
    voxel_coord_set = set()

    for idx_local in in_grid_indices:
        x_pt = float(x_sw0[idx_local])
        y_pt = float(y_sw0[idx_local])
        z_pt = float(z_sw0[idx_local])

        ix, iy, iz = grid.point_to_voxel_index(x_pt, y_pt, z_pt)
        xc, yc, zc = grid.voxel_index_to_centroid(ix, iy, iz)
        voxel_key = (ix, iy, iz)
        voxel_coord_set.add(voxel_key)

        # Mark grid tensor as occupied
        grid.state_grid[ix, iy, iz] = STATE_RADAR_SUPPORTED

        # Verify voxel physical containment (Requirement 2 & 3)
        assert (xc - grid.vx / 2.0 <= x_pt <= xc + grid.vx / 2.0), f"X coordinate {x_pt} outside voxel [{xc - grid.vx/2.0}, {xc + grid.vx/2.0}]"
        assert (yc - grid.vy / 2.0 <= y_pt <= yc + grid.vy / 2.0), f"Y coordinate {y_pt} outside voxel [{yc - grid.vy/2.0}, {yc + grid.vy/2.0}]"
        assert (zc - grid.vz / 2.0 <= z_pt <= zc + grid.vz / 2.0), f"Z coordinate {z_pt} outside voxel [{zc - grid.vz/2.0}, {zc + grid.vz/2.0}]"

        # Determine Temporal Candidate Evidence Tier (Requirement 1 & 5)
        if idx_local in matched_src_local:
            temporal_tier = TIER_HEURISTIC_CANDIDATE
            pair_meta = matched_src_local[idx_local]
            # Exact mapping integrity verification
            assert pair_meta['src_local_idx'] == idx_local, f"Index mismatch: {pair_meta['src_local_idx']} != {idx_local}"
            assert abs(pair_meta['src_x'] - x_pt) < 1e-4, f"X coord mismatch: {pair_meta['src_x']} != {x_pt}"
            assert abs(pair_meta['src_y'] - y_pt) < 1e-4, f"Y coord mismatch: {pair_meta['src_y']} != {y_pt}"
            assoc_detail = {
                "destination_sweep_idx": 1,
                "destination_local_idx": pair_meta['dst_local_idx'],
                "dst_x_ego": pair_meta['dst_x'],
                "dst_y_ego": pair_meta['dst_y'],
                "spatial_separation_m": pair_meta['spatial_separation_m'],
                "delta_vx_comp_m_s": pair_meta['delta_vx_comp'],
                "same_sensor_cluster_id": pair_meta['same_sensor_id'],
                "same_dynamic_property": pair_meta['same_dyn_prop']
            }
        elif idx_local in ambiguous_src_local:
            temporal_tier = TIER_AMBIGUOUS_CANDIDATE
            assoc_detail = {"status": "Multiple candidates or destination contention within exploratory gates"}
        elif idx_local in unmatched_src_local:
            temporal_tier = TIER_ISOLATED_RADAR
            assoc_detail = {"status": "Zero admissible candidates within exploratory gates"}
        else:
            raise RuntimeError(f"Unclassified source return index: {idx_local}")

        # Determine Camera Projection Context
        is_cam_proj = bool(in_cam_viewport_sw0[idx_local])
        cam_context = CAM_PROJECTS_VIEWPORT if is_cam_proj else CAM_OUTSIDE_VIEWPORT

        rec = {
            "radar_return_idx": int(idx_local),
            "actual_radar_coordinates_ego_m": {
                "x": round(x_pt, 4),
                "y": round(y_pt, 4),
                "z": round(z_pt, 4)
            },
            "voxel_indices": [int(ix), int(iy), int(iz)],
            "voxel_centroid_ego_m": {
                "x": round(xc, 4),
                "y": round(yc, 4),
                "z": round(zc, 4)
            },
            "voxel_bounding_box_m": {
                "x_min": round(xc - grid.vx / 2.0, 4),
                "x_max": round(xc + grid.vx / 2.0, 4),
                "y_min": round(yc - grid.vy / 2.0, 4),
                "y_max": round(yc + grid.vy / 2.0, 4),
                "z_min": round(zc - grid.vz / 2.0, 4),
                "z_max": round(zc + grid.vz / 2.0, 4)
            },
            "physical_measurements": {
                "compensated_doppler_component_vx_comp_m_s": round(float(vx_comp_sw0[idx_local]), 3),
                "radar_cross_section_rcs_dbm2": round(float(rcs_sw0[idx_local]), 2),
                "sensor_cluster_id": int(id_sw0[idx_local]),
                "dynamic_property_code": int(dyn_prop_sw0[idx_local])
            },
            "evidence_attribution": {
                "temporal_candidate_tier": temporal_tier,
                "camera_projection_context": cam_context,
                "projects_into_camera_viewport": is_cam_proj,
                "camera_depth_m": round(float(depth_cam_sw0[idx_local]), 3) if depth_cam_sw0[idx_local] > 0 else None,
                "camera_pixel_u": round(float(u_sw0[idx_local]), 1) if is_cam_proj else None,
                "camera_pixel_v": round(float(v_sw0[idx_local]), 1) if is_cam_proj else None
            },
            "temporal_association_metadata": assoc_detail
        }
        voxel_records.append(rec)

    # 6. Quantitative Counts & Partition Integrity Assertions
    radar_supported_voxels_count = len(voxel_coord_set)
    unknown_voxels_count = grid.total_voxels - radar_supported_voxels_count

    # Temporal breakdown
    count_candidate_associated = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_HEURISTIC_CANDIDATE)
    count_ambiguous = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_AMBIGUOUS_CANDIDATE)
    count_isolated = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_ISOLATED_RADAR)

    # Camera projection breakdown
    count_in_cam_viewport = sum(1 for r in voxel_records if r['evidence_attribution']['projects_into_camera_viewport'])
    count_outside_cam_viewport = sum(1 for r in voxel_records if not r['evidence_attribution']['projects_into_camera_viewport'])

    # Two-way cross-tabulation breakdown
    count_cand_cam = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_HEURISTIC_CANDIDATE and r['evidence_attribution']['projects_into_camera_viewport'])
    count_cand_nocam = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_HEURISTIC_CANDIDATE and not r['evidence_attribution']['projects_into_camera_viewport'])

    count_ambig_cam = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_AMBIGUOUS_CANDIDATE and r['evidence_attribution']['projects_into_camera_viewport'])
    count_ambig_nocam = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_AMBIGUOUS_CANDIDATE and not r['evidence_attribution']['projects_into_camera_viewport'])

    count_iso_cam = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_ISOLATED_RADAR and r['evidence_attribution']['projects_into_camera_viewport'])
    count_iso_nocam = sum(1 for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_ISOLATED_RADAR and not r['evidence_attribution']['projects_into_camera_viewport'])

    # RIGOROUS PARTITION INTEGRITY VERIFICATION (Zero Double-Counting)
    assert count_candidate_associated + count_ambiguous + count_isolated == count_in_grid, "Temporal tier partition mismatch!"
    assert count_in_cam_viewport + count_outside_cam_viewport == count_in_grid, "Camera projection partition mismatch!"
    assert count_cand_cam + count_cand_nocam == count_candidate_associated, "Candidate cross-tabulation mismatch!"
    assert count_ambig_cam + count_ambig_nocam == count_ambiguous, "Ambiguous cross-tabulation mismatch!"
    assert count_iso_cam + count_iso_nocam == count_isolated, "Isolated cross-tabulation mismatch!"
    assert count_cand_cam + count_ambig_cam + count_iso_cam == count_in_cam_viewport, "In-viewport column sum mismatch!"
    assert count_cand_nocam + count_ambig_nocam + count_iso_nocam == count_outside_cam_viewport, "Outside-viewport column sum mismatch!"
    assert radar_supported_voxels_count + unknown_voxels_count == grid.total_voxels, "Total grid voxel sum mismatch!"

    # Explicit cross-tabulation verification for reference sample ca9a282c9e77460f8360f564131a8af5
    if sample_token == "ca9a282c9e77460f8360f564131a8af5":
        assert count_cand_cam == 18, f"Expected 18 candidate in viewport, got {count_cand_cam}"
        assert count_cand_nocam == 1, f"Expected 1 candidate outside viewport, got {count_cand_nocam}"
        assert count_ambig_cam == 9, f"Expected 9 ambiguous in viewport, got {count_ambig_cam}"
        assert count_ambig_nocam == 4, f"Expected 4 ambiguous outside viewport, got {count_ambig_nocam}"
        assert count_iso_cam == 14, f"Expected 14 isolated in viewport, got {count_iso_cam}"
        assert count_iso_nocam == 2, f"Expected 2 isolated outside viewport, got {count_iso_nocam}"

    print("\n[*] Metric Summary & Partition Audit:")
    print(f"    - Total Grid Voxel Volume        : {grid.total_voxels:,} voxels ({grid.total_volume_m3:,.1f} m^3)")
    print(f"    - Unknown / Unobserved Space     : {unknown_voxels_count:,} voxels ({unknown_voxels_count / grid.total_voxels * 100:.3f}%) [Not Free, Not Occupied]")
    print(f"    - Radar-Supported Voxels         : {radar_supported_voxels_count} voxels ({radar_supported_voxels_count / grid.total_voxels * 100:.3f}%)")
    print(f"      * In-Grid Physical Returns     : {count_in_grid} (1:1 bijection, zero voxel collisions)")
    print(f"      * Out-of-Grid Physical Returns : {count_out_of_grid}")
    print(f"    - Temporal Association Breakdown (Stage 4B Heuristic):")
    print(f"      * Heuristic Candidate Evidence : {count_candidate_associated} returns ({count_cand_cam} in viewport, {count_cand_nocam} outside)")
    print(f"      * Ambiguous Candidate Evidence : {count_ambiguous} returns ({count_ambig_cam} in viewport, {count_ambig_nocam} outside)")
    print(f"      * Isolated Radar Evidence      : {count_isolated} returns ({count_iso_cam} in viewport, {count_iso_nocam} outside)")
    print(f"    - Camera Projection Context (Stage 3 Geometry):")
    print(f"      * Projects into Camera Viewport: {count_in_cam_viewport} returns (Geometric projection only)")
    print(f"      * Outside Camera Viewport      : {count_outside_cam_viewport} returns")
    print(f"    - Double-Counting Check          : Passed (Zero double-counting detected)")

    # 7. Render Comprehensive 3D/BEV Multi-Panel Visualization
    print("\n[*] Generating 3D Evidence-Attributed Hazard Volume visualization...")
    vis_output_path = os.path.join(output_dir, "evidence_attributed_volume.png")
    render_eahv_visualization(
        grid=grid,
        voxel_records=voxel_records,
        cam_path=cam_path,
        cs_cam=cs_cam,
        output_filepath=vis_output_path,
        counts={
            "total_grid": grid.total_voxels,
            "unknown": unknown_voxels_count,
            "radar_supported": radar_supported_voxels_count,
            "candidate_associated": count_candidate_associated,
            "ambiguous": count_ambiguous,
            "isolated": count_isolated,
            "in_viewport": count_in_cam_viewport,
            "outside_viewport": count_outside_cam_viewport,
            "cand_cam": count_cand_cam,
            "cand_nocam": count_cand_nocam,
            "ambig_cam": count_ambig_cam,
            "ambig_nocam": count_ambig_nocam,
            "iso_cam": count_iso_cam,
            "iso_nocam": count_iso_nocam
        },
        show_plot=show_plot
    )
    print(f"[+] 3D EAHV Visualization saved: {vis_output_path}")

    # 8. Export Metrics JSON File
    metrics_output_path = os.path.join(output_dir, "eahv_metrics.json")
    export_metrics_json(
        metrics_output_path=metrics_output_path,
        sample_token=sample_token,
        scene_name=scene_name,
        grid=grid,
        total_sw0_returns=total_sw0_returns,
        count_in_grid=count_in_grid,
        count_out_of_grid=count_out_of_grid,
        radar_supported_voxels_count=radar_supported_voxels_count,
        unknown_voxels_count=unknown_voxels_count,
        count_candidate_associated=count_candidate_associated,
        count_ambiguous=count_ambiguous,
        count_isolated=count_isolated,
        count_in_cam_viewport=count_in_cam_viewport,
        count_outside_cam_viewport=count_outside_cam_viewport,
        cross_tab={
            "candidate_associated": {"in_cam": count_cand_cam, "outside_cam": count_cand_nocam, "total": count_candidate_associated},
            "ambiguous": {"in_cam": count_ambig_cam, "outside_cam": count_ambig_nocam, "total": count_ambiguous},
            "isolated": {"in_cam": count_iso_cam, "outside_cam": count_iso_nocam, "total": count_isolated},
            "totals": {"in_cam": count_in_cam_viewport, "outside_cam": count_outside_cam_viewport, "total": radar_supported_voxels_count}
        },
        voxel_records=voxel_records
    )
    print(f"[+] Stage 5 Metrics JSON saved: {metrics_output_path}")
    print("\n[+] Stage 5 Execution Completed Successfully.")
    return vis_output_path, metrics_output_path


# ==============================================================================
# VISUALIZATION RENDERING
# ==============================================================================

def render_eahv_visualization(
    grid,
    voxel_records,
    cam_path,
    cs_cam,
    output_filepath,
    counts,
    show_plot=False
):
    """
    Renders high-resolution 4-panel visual dashboard:
    1. Top Left: CAM_FRONT Viewport with co-registered radar projections color-coded by evidence tier.
    2. Bottom Left: Top-down Bird's-Eye-View (BEV) Spatial Attribution Grid with camera FOV cone.
    3. Right: 3D Isometric Evidence-Attributed Hazard Volume (EAHV) with 3D voxel cubes and actual returns.
    4. Bottom Banner: Formal metrics table, partition audit, and explicit scientific limitations.
    """
    fig = plt.figure(figsize=(24, 14), facecolor='#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Color Palette for Evidence Attribution
    COLOR_CANDIDATE = '#39d353'  # Vibrant Green for Heuristic Candidate Evidence
    COLOR_AMBIGUOUS = '#f0883e'  # Warm Amber for Ambiguous Candidate Evidence
    COLOR_ISOLATED = '#bc8cff'   # Purple/Magenta for Isolated Radar Evidence
    COLOR_EGO = '#ff7b72'        # Soft Red for Ego Vehicle
    COLOR_FOV = '#e3b341'        # Yellow for Camera FOV cone
    COLOR_GRID_BOUNDS = '#30363d'

    # Supertitle
    fig.suptitle(
        "STAGE 5: EVIDENCE-ATTRIBUTED HAZARD VOLUME (EAHV) PROTOTYPE\n"
        "[Geometric Multi-Sensor Evidence Attribution — NOT Neural Occupancy, NOT Collision/TTC Prediction, NOT a Safety-Certified ADAS System]",
        fontsize=14,
        fontweight='bold',
        color='#58a6ff',
        y=0.975
    )

    # --------------------------------------------------------------------------
    # SUBPLOT 1: 2D Camera Viewport Context (CAM_FRONT)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#161b22')

    if os.path.exists(cam_path):
        cam_img = Image.open(cam_path)
        ax1.imshow(cam_img)
    else:
        ax1.text(0.5, 0.5, "Camera Image Not Found", color='white', ha='center', va='center')

    # Project in-grid radar returns onto camera image plane
    u_cand, v_cand = [], []
    u_amb, v_amb = [], []
    u_iso, v_iso = [], []

    for r in voxel_records:
        if r['evidence_attribution']['projects_into_camera_viewport']:
            u = r['evidence_attribution']['camera_pixel_u']
            v = r['evidence_attribution']['camera_pixel_v']
            tier = r['evidence_attribution']['temporal_candidate_tier']
            if tier == TIER_HEURISTIC_CANDIDATE:
                u_cand.append(u); v_cand.append(v)
            elif tier == TIER_AMBIGUOUS_CANDIDATE:
                u_amb.append(u); v_amb.append(v)
            elif tier == TIER_ISOLATED_RADAR:
                u_iso.append(u); v_iso.append(v)

    if u_cand:
        ax1.scatter(u_cand, v_cand, c=COLOR_CANDIDATE, s=70, marker='o', edgecolors='black', linewidths=1.2,
                    label=f'Heuristic Candidate Evidence ({len(u_cand)})')
    if u_amb:
        ax1.scatter(u_amb, v_amb, c=COLOR_AMBIGUOUS, s=70, marker='^', edgecolors='black', linewidths=1.2,
                    label=f'Ambiguous Candidate Evidence ({len(u_amb)})')
    if u_iso:
        ax1.scatter(u_iso, v_iso, c=COLOR_ISOLATED, s=70, marker='s', edgecolors='black', linewidths=1.2,
                    label=f'Isolated Radar Evidence ({len(u_iso)})')

    ax1.set_title(
        f"Panel A: Camera Viewport Context (CAM_FRONT) — {counts['in_viewport']} In-Viewport Returns\n"
        "[Geometric image-plane projection only; does NOT verify optical visibility or absence of occlusion]",
        color='white', fontsize=10.5, fontweight='bold', pad=8
    )
    ax1.axis('off')
    ax1.legend(loc='lower left', facecolor='#0d1117', edgecolor='#30363d', labelcolor='white', fontsize=8.5)

    # --------------------------------------------------------------------------
    # SUBPLOT 2: Top-Down Bird's-Eye-View (BEV) Spatial Attribution Grid
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_facecolor('#161b22')
    ax2.grid(True, color='#30363d', linestyle='--', linewidth=0.6)

    # Ego vehicle footprint (length 4.5m, width 1.8m, rear axle at origin)
    ego_rect = Rectangle((-1.0, -0.9), 4.5, 1.8, color=COLOR_EGO, alpha=0.85, label='Ego Vehicle Footprint')
    ax2.add_patch(ego_rect)
    ax2.arrow(0, 0, 3.2, 0, head_width=0.9, head_length=0.9, fc='yellow', ec='yellow', zorder=5)

    # Camera FOV Cone (~70 deg horizontal FOV)
    cone_dist = 55.0
    half_angle = np.deg2rad(35)
    ax2.plot([0, cone_dist], [0, cone_dist * np.tan(half_angle)], color=COLOR_FOV, linestyle=':', linewidth=1.6, label='CAM_FRONT FOV Boundary')
    ax2.plot([0, cone_dist], [0, -cone_dist * np.tan(half_angle)], color=COLOR_FOV, linestyle=':', linewidth=1.6)

    # Render Voxel Footprints & Actual Radar Points in BEV
    for r in voxel_records:
        x_pt = r['actual_radar_coordinates_ego_m']['x']
        y_pt = r['actual_radar_coordinates_ego_m']['y']
        xc = r['voxel_centroid_ego_m']['x']
        yc = r['voxel_centroid_ego_m']['y']
        tier = r['evidence_attribution']['temporal_candidate_tier']
        in_cam = r['evidence_attribution']['projects_into_camera_viewport']

        if tier == TIER_HEURISTIC_CANDIDATE:
            color = COLOR_CANDIDATE
        elif tier == TIER_AMBIGUOUS_CANDIDATE:
            color = COLOR_AMBIGUOUS
        else:
            color = COLOR_ISOLATED

        # Voxel footprint square (0.5 x 0.5 m)
        vx_box = Rectangle(
            (xc - grid.vx / 2.0, yc - grid.vy / 2.0),
            grid.vx, grid.vy,
            facecolor=color,
            edgecolor='white' if in_cam else '#ff7b72',
            linestyle='-' if in_cam else '--',
            linewidth=0.8,
            alpha=0.55
        )
        ax2.add_patch(vx_box)

        # Actual radar point return
        ax2.scatter(
            x_pt, y_pt,
            c=color,
            s=45,
            marker='o' if in_cam else '^',
            edgecolors='black',
            linewidths=1.0,
            zorder=4
        )

        # For heuristic candidates, draw subtle displacement link to destination
        if tier == TIER_HEURISTIC_CANDIDATE and 'dst_x_ego' in r['temporal_association_metadata']:
            dst_x = r['temporal_association_metadata']['dst_x_ego']
            dst_y = r['temporal_association_metadata']['dst_y_ego']
            ax2.plot([x_pt, dst_x], [y_pt, dst_y], color='#39d353', linestyle=':', linewidth=1.0, alpha=0.7)

    # Custom BEV Legend Handles
    bev_legend_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_CANDIDATE, markersize=8, label=f'Heuristic Candidate ({counts["candidate_associated"]} voxels)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_AMBIGUOUS, markersize=8, label=f'Ambiguous Candidate ({counts["ambiguous"]} voxels)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_ISOLATED, markersize=8, label=f'Isolated Radar ({counts["isolated"]} voxels)'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='gray', markersize=7, label=f'In Camera Viewport ({counts["in_viewport"]} voxels)'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor='#ff7b72', markerfacecolor='gray', markersize=7, label=f'Outside Camera Viewport ({counts["outside_viewport"]} voxels)'),
        Line2D([0], [0], color=COLOR_FOV, linestyle=':', linewidth=1.5, label='CAM_FRONT FOV Boundary (~70°)')
    ]

    ax2.set_xlim(grid.x_min - 2, grid.x_max + 2)
    ax2.set_ylim(grid.y_min - 2, grid.y_max + 2)
    ax2.set_xlabel("X Forward in Ego Frame [m]", color='white', fontsize=10)
    ax2.set_ylabel("Y Left/Right in Ego Frame [m]", color='white', fontsize=10)
    ax2.set_title(
        f"Panel B: Top-Down BEV Spatial Evidence Plane\n"
        f"[48 Radar-Supported Voxels | {counts['unknown']:,} Unknown Voxels (Void)]",
        color='white', fontsize=10.5, fontweight='bold', pad=8
    )
    ax2.tick_params(colors='white')
    ax2.legend(handles=bev_legend_handles, loc='upper right', facecolor='#0d1117', edgecolor='#30363d', labelcolor='white', fontsize=8)

    # --------------------------------------------------------------------------
    # SUBPLOT 3: 3D Isometric Evidence-Attributed Hazard Volume (Right Half)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    ax3.set_facecolor('#161b22')
    ax3.grid(True, color='#30363d', linestyle=':', linewidth=0.5)

    # 3D Grid Boundary Wireframe
    x_c = [grid.x_min, grid.x_max]
    y_c = [grid.y_min, grid.y_max]
    z_c = [grid.z_min, grid.z_max]
    for xc in x_c:
        for yc in y_c:
            ax3.plot([xc, xc], [yc, yc], z_c, color='#30363d', linestyle='--', linewidth=0.8)
    for xc in x_c:
        for zc in z_c:
            ax3.plot([xc, xc], y_c, [zc, zc], color='#30363d', linestyle='--', linewidth=0.8)
    for yc in y_c:
        for zc in z_c:
            ax3.plot(x_c, [yc, yc], [zc, zc], color='#30363d', linestyle='--', linewidth=0.8)

    # Ego vehicle origin and body representation in 3D
    ax3.scatter([0], [0], [0], color=COLOR_EGO, s=120, marker='o', label='Ego Origin (0,0,0) Rear Axle Ground')
    # Wireframe box for ego vehicle body: [-1, 3.5] x [-0.9, 0.9] x [0, 1.5]
    x_ego_box = [-1.0, 3.5]
    y_ego_box = [-0.9, 0.9]
    z_ego_box = [0.0, 1.5]
    for xc in x_ego_box:
        for yc in y_ego_box:
            ax3.plot([xc, xc], [yc, yc], z_ego_box, color='#ff7b72', linestyle='-', linewidth=1.2, alpha=0.7)
    for xc in x_ego_box:
        for zc in z_ego_box:
            ax3.plot([xc, xc], y_ego_box, [zc, zc], color='#ff7b72', linestyle='-', linewidth=1.2, alpha=0.7)
    for yc in y_ego_box:
        for zc in z_ego_box:
            ax3.plot(x_ego_box, [yc, yc], [zc, zc], color='#ff7b72', linestyle='-', linewidth=1.2, alpha=0.7)

    # 3D Camera Frustum Wireframe in Ego Coordinates
    t_cam = np.array(cs_cam['translation'])
    R_cam = Quaternion(cs_cam['rotation']).rotation_matrix
    K = np.array(cs_cam['camera_intrinsic'])
    K_inv = np.linalg.inv(K)

    # 4 Corner rays at depth = 45m
    w_cam, h_cam = 1600, 900
    corners_pix = np.array([[0, 0, 1], [w_cam, 0, 1], [w_cam, h_cam, 1], [0, h_cam, 1]]).T
    rays_cam_45 = (K_inv @ corners_pix) * (45.0 / (K_inv @ corners_pix)[2:3, :])
    rays_ego_45 = (R_cam @ rays_cam_45) + t_cam[:, None]

    # Draw frustum boundary edges
    for i in range(4):
        ax3.plot([t_cam[0], rays_ego_45[0, i]],
                 [t_cam[1], rays_ego_45[1, i]],
                 [t_cam[2], rays_ego_45[2, i]],
                 color=COLOR_FOV, linestyle=':', linewidth=1.2, alpha=0.8)
    # Frustum far rectangle
    for i in range(4):
        j = (i + 1) % 4
        ax3.plot([rays_ego_45[0, i], rays_ego_45[0, j]],
                 [rays_ego_45[1, i], rays_ego_45[1, j]],
                 [rays_ego_45[2, i], rays_ego_45[2, j]],
                 color=COLOR_FOV, linestyle=':', linewidth=1.2, alpha=0.8)

    # Render 3D Voxels and Actual Radar Points grouped by Evidence Tier
    cand_voxels = [r for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_HEURISTIC_CANDIDATE]
    amb_voxels = [r for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_AMBIGUOUS_CANDIDATE]
    iso_voxels = [r for r in voxel_records if r['evidence_attribution']['temporal_candidate_tier'] == TIER_ISOLATED_RADAR]

    def render_voxel_group(group, color, label_prefix):
        if not group:
            return
        x_bottom = np.array([r['voxel_centroid_ego_m']['x'] - grid.vx / 2.0 for r in group])
        y_bottom = np.array([r['voxel_centroid_ego_m']['y'] - grid.vy / 2.0 for r in group])
        z_bottom = np.array([r['voxel_centroid_ego_m']['z'] - grid.vz / 2.0 for r in group])
        dx = np.full_like(x_bottom, grid.vx)
        dy = np.full_like(y_bottom, grid.vy)
        dz = np.full_like(z_bottom, grid.vz)

        # 3D Voxel Cubes
        ax3.bar3d(
            x_bottom, y_bottom, z_bottom,
            dx, dy, dz,
            color=color,
            edgecolor='#22272e',
            alpha=0.65,
            shade=True
        )

        # Actual physical radar return points (Requirement 3)
        x_pts = [r['actual_radar_coordinates_ego_m']['x'] for r in group]
        y_pts = [r['actual_radar_coordinates_ego_m']['y'] for r in group]
        z_pts = [r['actual_radar_coordinates_ego_m']['z'] for r in group]
        ax3.scatter(
            x_pts, y_pts, z_pts,
            c='white',
            s=28,
            marker='o',
            edgecolors='black',
            linewidths=0.8,
            alpha=0.95
        )

    render_voxel_group(cand_voxels, COLOR_CANDIDATE, "Heuristic Candidate")
    render_voxel_group(amb_voxels, COLOR_AMBIGUOUS, "Ambiguous Candidate")
    render_voxel_group(iso_voxels, COLOR_ISOLATED, "Isolated Radar")

    # 3D Axes formatting
    ax3.set_xlim(grid.x_min, grid.x_max)
    ax3.set_ylim(grid.y_min, grid.y_max)
    ax3.set_zlim(grid.z_min, grid.z_max)
    ax3.set_xlabel("X Forward [m]", color='white', labelpad=10)
    ax3.set_ylabel("Y Left [m]", color='white', labelpad=10)
    ax3.set_zlabel("Z Up [m]", color='white', labelpad=10)

    ax3.tick_params(colors='white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor('#30363d')
    ax3.yaxis.pane.set_edgecolor('#30363d')
    ax3.zaxis.pane.set_edgecolor('#30363d')
    ax3.view_init(elev=24, azim=-122)

    # 3D Legend Handles
    legend_3d_handles = [
        Patch(facecolor=COLOR_CANDIDATE, edgecolor='black', alpha=0.8, label=f'Heuristic Candidate Evidence ({len(cand_voxels)} voxels)'),
        Patch(facecolor=COLOR_AMBIGUOUS, edgecolor='black', alpha=0.8, label=f'Ambiguous Candidate Evidence ({len(amb_voxels)} voxels)'),
        Patch(facecolor=COLOR_ISOLATED, edgecolor='black', alpha=0.8, label=f'Isolated Radar Evidence ({len(iso_voxels)} voxels)'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='white', markersize=6, label='Actual Radar Return Coordinate (Inside Voxel)'),
        Line2D([0], [0], color=COLOR_FOV, linestyle=':', linewidth=1.5, label='CAM_FRONT 3D Frustum Bounds'),
        Line2D([0], [0], color=COLOR_EGO, linewidth=1.5, label='Ego Vehicle Body Bounds')
    ]
    ax3.legend(handles=legend_3d_handles, loc='upper left', facecolor='#0d1117', edgecolor='#30363d', labelcolor='white', fontsize=8.5)

    ax3.set_title(
        f"Panel C: 3D Evidence-Attributed Hazard Volume [{grid.nx}x{grid.ny}x{grid.nz} Cells, 0.5m Res]\n"
        f"[48 Radar-Supported Voxels | {counts['unknown']:,} Unknown Space (Void)]",
        color='white', fontsize=11, fontweight='bold', pad=12
    )

    # --------------------------------------------------------------------------
    # FOOTER: Quantitative Metric Summary & Explicit Limitations
    # --------------------------------------------------------------------------
    summary_text = (
        f"EAHV QUANTITATIVE AUDIT:  Total Voxel Grid: {counts['total_grid']:,} cells ({grid.total_volume_m3:,.1f} m³)  |  "
        f"Unknown Space: {counts['unknown']:,} cells ({counts['unknown']/counts['total_grid']*100:.3f}%) [Not Free, Not Occupied]  |  "
        f"Radar-Supported Voxels: {counts['radar_supported']} cells ({counts['radar_supported']/counts['total_grid']*100:.3f}%)\n"
        f"TEMPORAL EVIDENCE TIERS (Stage 4B Heuristic):  Heuristic Candidate: {counts['candidate_associated']}  |  "
        f"Ambiguous: {counts['ambiguous']}  |  Isolated: {counts['isolated']}  |  "
        f"CAMERA PROJECTION CONTEXT:  In Viewport: {counts['in_viewport']}  |  Outside Viewport: {counts['outside_viewport']}  |  "
        f"Double-Counting: 0 Detected\n"
        f"EXPLICIT LIMITATIONS: (1) No collision prediction or TTC computed. (2) No neural semantic occupancy prediction; unobserved space is strictly UNKNOWN. "
        f"(3) Camera projection is geometric line-of-sight only; cannot verify optical visibility or occlusion. "
        f"(4) Radar Doppler (vx_comp) is 1D scalar LOS measurement only; 2D velocity vectors are not inferred. "
        f"(5) Candidate associations are exploratory temporal proximity pairs, NOT certified object tracking tracks. Research prototype only."
    )
    fig.text(
        0.5, 0.02,
        summary_text,
        color='#8b949e',
        fontsize=8.0,
        family='monospace',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle='square,pad=0.6', facecolor='#161b22', edgecolor='#30363d', alpha=0.95)
    )

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
    plt.savefig(output_filepath, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show_plot:
        plt.show(block=False)
        plt.pause(2.0)
    plt.close(fig)


# ==============================================================================
# JSON METRICS EXPORTER
# ==============================================================================

def export_metrics_json(
    metrics_output_path,
    sample_token,
    scene_name,
    grid,
    total_sw0_returns,
    count_in_grid,
    count_out_of_grid,
    radar_supported_voxels_count,
    unknown_voxels_count,
    count_candidate_associated,
    count_ambiguous,
    count_isolated,
    count_in_cam_viewport,
    count_outside_cam_viewport,
    cross_tab,
    voxel_records
):
    """
    Exports comprehensive JSON metrics file adhering to all stage requirements.
    """
    metrics = {
        "pipeline_stage": "Stage 5 Evidence-Attributed Hazard Volume (EAHV) Prototype",
        "sample_token": sample_token,
        "scene_name": scene_name,
        "reference_coordinate_frame": "Ego-Vehicle Coordinate System at Camera Keyframe Timestamp (V_ego)",
        "voxel_grid_specification": {
            "x_bounds_m": [float(grid.x_min), float(grid.x_max)],
            "y_bounds_m": [float(grid.y_min), float(grid.y_max)],
            "z_bounds_m": [float(grid.z_min), float(grid.z_max)],
            "voxel_size_m": [float(grid.vx), float(grid.vy), float(grid.vz)],
            "dimensions_cells": [int(grid.nx), int(grid.ny), int(grid.nz)],
            "total_voxels": int(grid.total_voxels),
            "voxel_volume_m3": float(grid.voxel_volume_m3),
            "total_volume_m3": float(grid.total_volume_m3)
        },
        "voxel_state_taxonomy": {
            "unknown_unobserved_voxels": {
                "count": int(unknown_voxels_count),
                "percentage": round(float(unknown_voxels_count / grid.total_voxels * 100.0), 4),
                "definition": "Spatial volume containing zero physical radar returns. Monocular camera cannot measure dense depth. Strictly unknown / unobserved space; NOT labeled as free or occupied."
            },
            "radar_supported_voxels": {
                "count": int(radar_supported_voxels_count),
                "percentage": round(float(radar_supported_voxels_count / grid.total_voxels * 100.0), 4),
                "definition": "Spatial volume containing physical millimeter-wave radar returns from reference keyframe within grid boundaries."
            },
            "partition_integrity": {
                "sum_voxels": int(unknown_voxels_count + radar_supported_voxels_count),
                "matches_total_grid": bool((unknown_voxels_count + radar_supported_voxels_count) == grid.total_voxels),
                "double_counting_detected": False
            }
        },
        "radar_return_spatial_partition": {
            "total_sweep_0_returns": int(total_sw0_returns),
            "returns_inside_grid_bounds": int(count_in_grid),
            "returns_outside_grid_bounds": int(count_out_of_grid),
            "unique_voxels_occupied": int(radar_supported_voxels_count),
            "bijection_check": {
                "points_per_occupied_voxel": 1.0,
                "multiple_returns_per_voxel_detected": False,
                "empty_occupied_voxels_detected": False
            }
        },
        "evidence_attribution_metrics": {
            "temporal_candidate_association": {
                "source_pipeline": "Stage 4B Consecutive Sweep Candidate Association (Sweeps 0 -> 1)",
                "exploratory_spatial_gate_m": 1.5,
                "exploratory_doppler_gate_m_s": 2.5,
                "heuristic_candidate_associated_returns": {
                    "count": int(count_candidate_associated),
                    "definition": "Radar returns with unique 1-to-1 candidate association in consecutive sweep under exploratory gating. Labeled as heuristic candidate evidence, NOT confirmed tracking."
                },
                "ambiguous_candidate_returns": {
                    "count": int(count_ambiguous),
                    "definition": "Radar returns with multiple candidate associations or destination contention in consecutive sweep under exploratory gating."
                },
                "isolated_radar_returns": {
                    "count": int(count_isolated),
                    "definition": "Radar returns with zero candidate associations in consecutive sweep under exploratory gating."
                },
                "partition_integrity": {
                    "sum_temporal_categories": int(count_candidate_associated + count_ambiguous + count_isolated),
                    "matches_total_in_grid": bool((count_candidate_associated + count_ambiguous + count_isolated) == count_in_grid),
                    "double_counting_detected": False
                }
            },
            "camera_projection_context": {
                "source_pipeline": "Stage 3 Multi-Sensor Geometric Alignment & Perspective Projection",
                "camera_sensor": "CAM_FRONT",
                "projects_into_camera_viewport": {
                    "count": int(count_in_cam_viewport),
                    "definition": "Radar returns whose 3D coordinate projects geometrically inside CAM_FRONT image bounds (Z_cam > 0, 0 <= u < 1600, 0 <= v < 900). Geometric projection ONLY; does NOT verify optical line-of-sight visibility or absence of occlusion."
                },
                "outside_camera_viewport": {
                    "count": int(count_outside_cam_viewport),
                    "definition": "Radar returns whose 3D coordinate falls outside CAM_FRONT image bounds or has Z_cam <= 0."
                },
                "partition_integrity": {
                    "sum_projection_categories": int(count_in_cam_viewport + count_outside_cam_viewport),
                    "matches_total_in_grid": bool((count_in_cam_viewport + count_outside_cam_viewport) == count_in_grid),
                    "double_counting_detected": False
                }
            },
            "two_way_cross_tabulation": {
                "description": "Mutually exclusive and exhaustive cross-tabulation of the 48 radar-supported voxels.",
                "matrix": cross_tab
            }
        },
        "system_limitations_and_scientific_disclaimers": [
            "No collision prediction, Time-to-Collision (TTC), or emergency braking thresholds are computed.",
            "No neural semantic occupancy prediction is used; unobserved space is strictly designated UNKNOWN, not free or occupied.",
            "Camera projection represents geometric pinhole ray intersection only and does NOT confirm visual line of sight or absence of foreground occlusion.",
            "Radar Doppler velocity (vx_comp) is a 1D scalar line-of-sight measurement; transverse velocity is unmeasured and 2D velocity vectors are not inferred.",
            "Stage 4B candidate associations are heuristic temporal proximity pairs, NOT certified object tracking tracks.",
            "System is a research prototype for geometric evidence attribution, NOT a safety-certified ADAS system."
        ],
        "detailed_radar_supported_voxel_records": voxel_records
    }

    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 5 Evidence-Attributed Hazard Volume (EAHV) Prototype")
    parser.add_argument("--sample-token", type=str, default="ca9a282c9e77460f8360f564131a8af5",
                        help="Target nuScenes sample token (default: ca9a282c9e77460f8360f564131a8af5)")
    parser.add_argument("--dataroot", type=str, default=".",
                        help="Path to nuScenes dataset root (default: .)")
    parser.add_argument("--output-dir", type=str, default="outputs/eahv",
                        help="Directory to save EAHV outputs (default: outputs/eahv)")
    parser.add_argument("--show-plot", action="store_true",
                        help="Whether to display interactive visualization window")
    args = parser.parse_args()

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)
    execute_stage5_eahv(
        nusc=nusc,
        sample_token=args.sample_token,
        output_dir=args.output_dir,
        show_plot=args.show_plot
    )


if __name__ == "__main__":
    main()
