#!/usr/bin/env python3
"""
Stage 1 Baseline: Radar-Return 3D Voxelisation
===============================================
Project: Radar–Camera Fusion-Based Driver Assistance System (MY_ADAS_PROJECT)
Author / Pipeline: Stage 1 Geometric Baseline

DISCLAIMER & LABELING:
----------------------
This script performs discrete physical sensor voxelisation of actual millimeter-wave
radar returns onto a 3D spatial grid. It is NOT a neural-network semantic occupancy
prediction model. Voxels marked as occupied represent observed discrete radar returns;
unmarked voxels represent unknown / unobserved space.

COORDINATE SYSTEM SPECIFICATION:
--------------------------------
Target Coordinate Frame: Ego-Vehicle Coordinate System at Camera Keyframe Timestamp (V_ego)
- Origin (0, 0, 0): Center of the vehicle's rear axle projected onto the road ground surface.
- +X axis: Longitudinal forward along the vehicle heading.
- +Y axis: Lateral leftward (perpendicular to heading).
- +Z axis: Vertical upward (normal to the ground plane).

Sensor Transformation Sequence (rigorous nuScenes calibration chain):
1. P_radar (Sensor Frame S_radar) -> P_ego_radar (Ego Frame at Radar timestamp)
   using calibrated_sensor(RADAR_FRONT)['rotation' and 'translation']
2. P_ego_radar -> P_global (Global World Frame W_world)
   using ego_pose(RADAR_FRONT)['rotation' and 'translation']
3. P_global -> P_ego_cam (Ego Frame at Camera timestamp V_ego)
   using inverse ego_pose(CAM_FRONT)['rotation' and 'translation']
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
# Use TkAgg or default GUI backend, with headless fallback if needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points


# Voxel State Definitions
STATE_UNKNOWN = 0          # Unobserved / unknown free space
STATE_RADAR_OCCUPIED = 1   # Observed physical radar return


class VoxelGrid3D:
    """
    3D Regular Voxel Grid in Ego-Vehicle Coordinates.
    
    Spatial bounds:
      X in [x_min, x_max] (Forward, meters)
      Y in [y_min, y_max] (Lateral Left/Right, meters)
      Z in [z_min, z_max] (Elevation Up/Down, meters)
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

        # 3D occupancy grid tensor (uint8)
        self.occupancy_grid = np.full(self.grid_dimensions, STATE_UNKNOWN, dtype=np.uint8)

        # Statistical channels per voxel
        self.count_grid = np.zeros(self.grid_dimensions, dtype=np.int32)
        self.vx_grid = np.zeros(self.grid_dimensions, dtype=np.float32)
        self.vy_grid = np.zeros(self.grid_dimensions, dtype=np.float32)
        self.rcs_grid = np.zeros(self.grid_dimensions, dtype=np.float32)

    def point_to_voxel_index(self, x, y, z):
        """Converts physical continuous coordinates (meters) to integer voxel indices."""
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
        """Filter mask for points falling inside the defined voxel boundaries."""
        return (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max) &
            (z >= self.z_min) & (z < self.z_max)
        )

    def voxelize_radar_points(self, points):
        """
        Populate the 3D grid with transformed radar points.
        points: (18, N) array where:
          points[0] = x (forward, ego)
          points[1] = y (left, ego)
          points[2] = z (up, ego)
          points[5] = rcs (Radar Cross Section)
          points[8] = vx_comp (longitudinal compensated velocity in ego)
          points[9] = vy_comp (lateral compensated velocity in ego)
        """
        x = points[0, :]
        y = points[1, :]
        z = points[2, :]

        rcs = points[5, :] if points.shape[0] > 5 else np.zeros_like(x)
        vx = points[8, :] if points.shape[0] > 8 else (points[6, :] if points.shape[0] > 6 else np.zeros_like(x))
        vy = points[9, :] if points.shape[0] > 9 else (points[7, :] if points.shape[0] > 7 else np.zeros_like(x))

        mask = self.is_inside(x, y, z)
        valid_x = x[mask]
        valid_y = y[mask]
        valid_z = z[mask]
        valid_vx = vx[mask]
        valid_vy = vy[mask]
        valid_rcs = rcs[mask]

        ix, iy, iz = self.point_to_voxel_index(valid_x, valid_y, valid_z)

        occupied_coords = []
        for i in range(len(ix)):
            cur_ix, cur_iy, cur_iz = ix[i], iy[i], iz[i]
            self.occupancy_grid[cur_ix, cur_iy, cur_iz] = STATE_RADAR_OCCUPIED
            self.count_grid[cur_ix, cur_iy, cur_iz] += 1
            self.vx_grid[cur_ix, cur_iy, cur_iz] += valid_vx[i]
            self.vy_grid[cur_ix, cur_iy, cur_iz] += valid_vy[i]
            self.rcs_grid[cur_ix, cur_iy, cur_iz] += valid_rcs[i]
            occupied_coords.append((cur_ix, cur_iy, cur_iz))

        # Average multi-point voxel returns
        num_occupied = int(np.sum(self.occupancy_grid == STATE_RADAR_OCCUPIED))
        nz_mask = self.count_grid > 0
        self.vx_grid[nz_mask] /= self.count_grid[nz_mask]
        self.vy_grid[nz_mask] /= self.count_grid[nz_mask]
        self.rcs_grid[nz_mask] /= self.count_grid[nz_mask]

        return num_occupied, valid_x, valid_y, valid_z, valid_vx, valid_vy


def transform_radar_to_ego_cam(nusc, sample):
    """
    Applies the full rigorous nuScenes coordinate transformation:
    RADAR_FRONT sensor frame -> Global frame -> Ego-Vehicle frame at CAM_FRONT timestamp.
    """
    cam_token = sample['data']['CAM_FRONT']
    radar_token = sample['data']['RADAR_FRONT']

    cam_data = nusc.get('sample_data', cam_token)
    radar_data = nusc.get('sample_data', radar_token)

    radar_path = os.path.join(nusc.dataroot, radar_data['filename'])
    cam_path = os.path.join(nusc.dataroot, cam_data['filename'])

    # Load raw point cloud
    pc = RadarPointCloud.from_file(radar_path)

    # 1. Rotate & Translate: Radar sensor -> Ego frame at radar sweep timestamp
    cs_radar = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_radar['rotation']).rotation_matrix)
    pc.translate(np.array(cs_radar['translation']))

    # 2. Rotate & Translate: Ego frame at radar timestamp -> Global frame
    pose_radar = nusc.get('ego_pose', radar_data['ego_pose_token'])
    pc.rotate(Quaternion(pose_radar['rotation']).rotation_matrix)
    pc.translate(np.array(pose_radar['translation']))

    # 3. Translate & Rotate: Global frame -> Ego frame at camera keyframe timestamp
    pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
    pc.translate(-np.array(pose_cam['translation']))
    pc.rotate(Quaternion(pose_cam['rotation']).rotation_matrix.T)

    return pc, cam_path, cam_data


def project_points_to_camera(nusc, cam_data, points_ego):
    """
    Projects 3D points from the Ego-Vehicle frame (at camera timestamp)
    onto the 2D CAM_FRONT image plane.
    """
    cs_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    # Points in ego frame -> camera sensor frame
    pts_cam = points_ego[:3, :].copy()
    pts_cam -= np.array(cs_cam['translation'])[:, None]
    pts_cam = np.dot(Quaternion(cs_cam['rotation']).rotation_matrix.T, pts_cam)

    # Depth is distance along camera optical axis (Z in camera frame)
    depths = pts_cam[2, :]

    # Project via camera intrinsic matrix
    camera_intrinsic = np.array(cs_cam['camera_intrinsic'])
    pts_2d = view_points(pts_cam, camera_intrinsic, normalize=True)

    return pts_2d, depths


def render_baseline_visualization(cam_img,
                                  pts_2d,
                                  depths,
                                  voxel_grid,
                                  sample_token,
                                  scene_name,
                                  output_filepath,
                                  show_plot=False):
    """
    Generates a multi-panel visual dashboard showing:
    - 2D Camera Front image with projected radar return overlays.
    - Top-Down Bird's Eye View (BEV) of the 3D Voxel Grid and ego vehicle.
    - 3D Isometric View of the radar-occupied voxels in the ego vehicle frame.
    """
    img_w, img_h = cam_img.size

    # Filter 2D points inside image view with positive depth (> 1.0m)
    fov_mask = (depths > 1.0) & (pts_2d[0, :] >= 0) & (pts_2d[0, :] < img_w) & (pts_2d[1, :] >= 0) & (pts_2d[1, :] < img_h)
    u_visible = pts_2d[0, fov_mask]
    v_visible = pts_2d[1, fov_mask]
    depth_visible = depths[fov_mask]

    # Extract occupied voxel indices and physical centroid coordinates
    occ_indices = np.argwhere(voxel_grid.occupancy_grid == STATE_RADAR_OCCUPIED)
    num_occ = len(occ_indices)

    occ_x = []
    occ_y = []
    occ_z = []
    occ_vel = []

    for ix, iy, iz in occ_indices:
        xc, yc, zc = voxel_grid.voxel_index_to_centroid(ix, iy, iz)
        occ_x.append(xc)
        occ_y.append(yc)
        occ_z.append(zc)
        # Doppler speed magnitude in horizontal plane
        spd = np.hypot(voxel_grid.vx_grid[ix, iy, iz], voxel_grid.vy_grid[ix, iy, iz])
        occ_vel.append(spd)

    occ_x = np.array(occ_x)
    occ_y = np.array(occ_y)
    occ_z = np.array(occ_z)
    occ_vel = np.array(occ_vel)

    # Setup high-resolution figure (18 x 9.5 inches)
    fig = plt.figure(figsize=(18, 9.5), dpi=120, facecolor='#111625')
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.96, wspace=0.25, hspace=0.3)

    # Main Title Header with required disclaimer
    fig.suptitle(
        "RADAR-RETURN 3D VOXELISATION BASELINE (STAGE 1)\n"
        "[DISCLAIMER: Sensor-Return Physical Voxelisation — NOT Neural-Network Semantic Occupancy Prediction]",
        fontsize=14,
        fontweight='bold',
        color='#00e5ff',
        y=0.96
    )

    # ----------------------------------------------------
    # Subplot 1: Camera Front View with Radar Projections
    # ----------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#0d1117')
    ax1.imshow(cam_img)
    if len(u_visible) > 0:
        scatter = ax1.scatter(
            u_visible, v_visible,
            c=depth_visible,
            cmap='plasma',
            s=80,
            edgecolors='cyan',
            linewidths=1.5,
            alpha=0.9,
            label='Projected Radar Return'
        )
        cbar = fig.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.03)
        cbar.set_label('Radar Depth (m)', color='white', fontsize=9)
        cbar.ax.tick_params(colors='white')
    ax1.set_title(
        f"Synchronized CAM_FRONT Image (Overlay: {len(u_visible)} Projected Returns)",
        color='white',
        fontsize=11,
        fontweight='bold'
    )
    ax1.axis('off')

    # ----------------------------------------------------
    # Subplot 2: Top-Down Bird's Eye View (BEV) Voxel Grid
    # ----------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_facecolor('#0d1117')
    ax2.grid(True, color='#21262d', linestyle='--', linewidth=0.8)

    # Draw Ego Vehicle Footprint at (0, 0) in Ego Frame
    # NuScenes typical car footprint: ~4.5m long, 1.8m wide, rear axle at (0, 0)
    ego_rect = plt.Rectangle(
        (-1.0, -0.9), 4.5, 1.8,
        fill=True, color='#ff0055', alpha=0.8,
        label='Ego Vehicle (Rear Axle Origin)'
    )
    ax2.add_patch(ego_rect)
    ax2.arrow(0, 0, 3.0, 0, head_width=0.8, head_length=0.8, fc='yellow', ec='yellow', label='+X Heading')

    # Draw CAM_FRONT approximate field of view cone (~70 deg FOV)
    cone_dist = 50.0
    half_angle = np.deg2rad(35)
    ax2.plot([0, cone_dist], [0, cone_dist * np.tan(half_angle)], color='#ffea00', linestyle=':', linewidth=1.2, label='CAM_FRONT FOV Cone')
    ax2.plot([0, cone_dist], [0, -cone_dist * np.tan(half_angle)], color='#ffea00', linestyle=':', linewidth=1.2)

    # Plot occupied voxels in BEV
    if num_occ > 0:
        bev_scatter = ax2.scatter(
            occ_x, occ_y,
            c=occ_z,
            cmap='turbo',
            s=70,
            marker='s',
            edgecolors='white',
            linewidths=0.8,
            alpha=0.9,
            label=f'Radar Voxel Return ({num_occ} occupied)'
        )
        cbar2 = fig.colorbar(bev_scatter, ax=ax2, fraction=0.046, pad=0.03)
        cbar2.set_label('Voxel Height Z (m)', color='white', fontsize=9)
        cbar2.ax.tick_params(colors='white')

    ax2.set_xlim(voxel_grid.x_min - 2, voxel_grid.x_max + 2)
    ax2.set_ylim(voxel_grid.y_min - 2, voxel_grid.y_max + 2)
    ax2.set_xlabel("Longitudinal Distance X Forward (m)", color='white', fontsize=10)
    ax2.set_ylabel("Lateral Distance Y Left/Right (m)", color='white', fontsize=10)
    ax2.set_title("Top-Down BEV Voxel Grid (Ego Coordinates)", color='white', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=8)

    # ----------------------------------------------------
    # Subplot 3: 3D Isometric View of the Voxel Grid
    # ----------------------------------------------------
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    ax3.set_facecolor('#0d1117')
    ax3.grid(True, color='#21262d', linestyle=':', linewidth=0.6)

    # Draw 3D Voxel Bounding Box Wireframe
    x_corners = [voxel_grid.x_min, voxel_grid.x_max]
    y_corners = [voxel_grid.y_min, voxel_grid.y_max]
    z_corners = [voxel_grid.z_min, voxel_grid.z_max]
    for xc in x_corners:
        for yc in y_corners:
            ax3.plot([xc, xc], [yc, yc], z_corners, color='#30363d', linestyle='--', linewidth=0.8)
    for xc in x_corners:
        for zc in z_corners:
            ax3.plot([xc, xc], y_corners, [zc, zc], color='#30363d', linestyle='--', linewidth=0.8)
    for yc in y_corners:
        for zc in z_corners:
            ax3.plot(x_corners, [yc, yc], [zc, zc], color='#30363d', linestyle='--', linewidth=0.8)

    # Mark Ego Vehicle Origin (0, 0, 0)
    ax3.scatter([0], [0], [0], color='#ff0055', s=120, marker='o', label='Ego Origin (0,0,0)')

    # Render physical 3D voxel cubes using bar3d
    if num_occ > 0:
        # Physical lower-left corner of each occupied voxel
        corner_x = occ_x - voxel_grid.vx / 2.0
        corner_y = occ_y - voxel_grid.vy / 2.0
        corner_z = occ_z - voxel_grid.vz / 2.0

        ax3.bar3d(
            corner_x, corner_y, corner_z,
            voxel_grid.vx, voxel_grid.vy, voxel_grid.vz,
            color='#00e5ff',
            edgecolor='#0077b6',
            alpha=0.75,
            shade=True
        )

    ax3.set_xlim(voxel_grid.x_min, voxel_grid.x_max)
    ax3.set_ylim(voxel_grid.y_min, voxel_grid.y_max)
    ax3.set_zlim(voxel_grid.z_min, voxel_grid.z_max)

    ax3.set_xlabel("X Forward (m)", color='white', labelpad=8)
    ax3.set_ylabel("Y Left (m)", color='white', labelpad=8)
    ax3.set_zlabel("Z Elevation (m)", color='white', labelpad=8)

    ax3.set_title(
        f"3D Voxel Space [{voxel_grid.nx}x{voxel_grid.ny}x{voxel_grid.nz} voxels]\n"
        f"({num_occ} Observed Returns | Unoccupied = Unknown Space)",
        color='white',
        fontsize=11,
        fontweight='bold',
        pad=12
    )

    ax3.tick_params(colors='white')
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor('#21262d')
    ax3.yaxis.pane.set_edgecolor('#21262d')
    ax3.zaxis.pane.set_edgecolor('#21262d')
    ax3.view_init(elev=26, azim=-125)

    # Save to disk
    plt.savefig(output_filepath, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show_plot:
        plt.show(block=False)
        plt.pause(2.0)
    plt.close(fig)


def run_voxel_baseline(sample_idx=0,
                       scene_idx=None,
                       voxel_size=(0.5, 0.5, 0.5),
                       output_dir="outputs/baseline",
                       show_plot=False):
    """
    Main execution pipeline for Stage 1 Baseline.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print(" STAGE 1 BASELINE: RADAR-RETURN 3D VOXELISATION")
    print(" (DISCLAIMER: Sensor Point Voxelisation, NOT Semantic Occupancy)")
    print("=================================================================")

    # 1. Load NuScenes mini dataset
    print(f"[*] Initializing nuScenes v1.0-mini from local workspace...")
    nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)

    # Determine target sample
    if scene_idx is not None:
        target_scene = nusc.scene[scene_idx]
        first_token = target_scene['first_sample_token']
        sample = nusc.get('sample', first_token)
        scene_name = target_scene['name']
        scene_desc = target_scene['description']
    else:
        sample = nusc.sample[sample_idx]
        scene_record = nusc.get('scene', sample['scene_token'])
        scene_name = scene_record['name']
        scene_desc = scene_record['description']

    sample_token = sample['token']
    print(f"[*] Target Sample Token : {sample_token}")
    print(f"[*] Scene Name          : {scene_name} ({scene_desc})")

    # 2. Coordinate Transformation (Radar -> Global -> Ego at Cam Time)
    print(f"[*] Transforming radar points using calibrated_sensor & ego_pose...")
    pc_ego, cam_path, cam_data = transform_radar_to_ego_cam(nusc, sample)
    total_raw_points = pc_ego.points.shape[1]
    print(f"[*] Raw Radar Points Loaded : {total_raw_points}")

    # 3. Create 3D Voxel Grid & Voxelize Radar Returns
    print(f"[*] Creating 3D Voxel Grid in Ego-Vehicle Coordinates...")
    voxel_grid = VoxelGrid3D(
        x_bounds=(0.0, 60.0),      # Forward: 0 to 60m
        y_bounds=(-25.0, 25.0),    # Lateral: -25 to +25m
        z_bounds=(-2.0, 4.0),      # Elevation: -2 to +4m
        voxel_size=voxel_size
    )

    num_occupied, valid_x, valid_y, valid_z, valid_vx, valid_vy = voxel_grid.voxelize_radar_points(pc_ego.points)

    # 4. Camera Projection
    cam_img = Image.open(cam_path)
    pts_2d, depths = project_points_to_camera(nusc, cam_data, pc_ego.points)

    # 5. Summary Reporting (Requirements 7 & 8)
    occupancy_ratio = (num_occupied / voxel_grid.total_voxels) * 100.0

    print("-----------------------------------------------------------------")
    print(f" METRIC SUMMARY & VOXEL SPECIFICATIONS:")
    print(f"-----------------------------------------------------------------")
    print(f" Sample Token        : {sample_token}")
    print(f" Coordinate Frame    : Ego-Vehicle Frame at Camera Timestamp (V_ego)")
    print(f" Spatial Boundaries  : X in [{voxel_grid.x_min}, {voxel_grid.x_max}] m (Forward)")
    print(f"                       Y in [{voxel_grid.y_min}, {voxel_grid.y_max}] m (Lateral)")
    print(f"                       Z in [{voxel_grid.z_min}, {voxel_grid.z_max}] m (Elevation)")
    print(f" Voxel Resolution    : dx={voxel_grid.vx}m, dy={voxel_grid.vy}m, dz={voxel_grid.vz}m")
    print(f" Grid Dimensions     : {voxel_grid.nx} x {voxel_grid.ny} x {voxel_grid.nz} (Nx x Ny x Nz)")
    print(f" Total Voxels        : {voxel_grid.total_voxels:,}")
    print(f" Occupied Voxels     : {num_occupied} (with verified radar returns)")
    print(f" Unobserved/Unknown  : {voxel_grid.total_voxels - num_occupied:,} voxels")
    print(f" Occupancy Ratio     : {occupancy_ratio:.5f}% (sparse radar return density)")
    print("-----------------------------------------------------------------")

    # 6. Save Metrics to JSON
    metrics_path = os.path.join(output_dir, "voxel_baseline_metrics.json")
    metrics_data = {
        "pipeline_stage": "Stage 1 Baseline",
        "description": "Discrete Radar-Return 3D Voxelisation (Non-Neural)",
        "sample_token": sample_token,
        "scene_name": scene_name,
        "scene_description": scene_desc,
        "coordinate_frame": "Ego-Vehicle Coordinate System at Camera Keyframe Timestamp (V_ego)",
        "grid_dimensions": {
            "nx": voxel_grid.nx,
            "ny": voxel_grid.ny,
            "nz": voxel_grid.nz
        },
        "voxel_resolution_m": {
            "dx": voxel_grid.vx,
            "dy": voxel_grid.vy,
            "dz": voxel_grid.vz
        },
        "spatial_bounds_m": {
            "x": [voxel_grid.x_min, voxel_grid.x_max],
            "y": [voxel_grid.y_min, voxel_grid.y_max],
            "z": [voxel_grid.z_min, voxel_grid.z_max]
        },
        "total_voxels": voxel_grid.total_voxels,
        "occupied_voxels": num_occupied,
        "unknown_unobserved_voxels": voxel_grid.total_voxels - num_occupied,
        "occupancy_percentage": occupancy_ratio,
        "raw_radar_points": total_raw_points,
        "in_roi_points": len(valid_x)
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[*] Metrics saved to   : {os.path.abspath(metrics_path)}")

    # 7. Render and Save Visualisation
    output_png = os.path.join(output_dir, "radar_voxel_baseline.png")
    print(f"[*] Generating 3D visualisation dashboard...")
    render_baseline_visualization(
        cam_img=cam_img,
        pts_2d=pts_2d,
        depths=depths,
        voxel_grid=voxel_grid,
        sample_token=sample_token,
        scene_name=scene_name,
        output_filepath=output_png,
        show_plot=show_plot
    )
    print(f"[*] Visualisation saved: {os.path.abspath(output_png)}")
    print("=================================================================")
    print(" STAGE 1 BASELINE COMPLETED SUCCESSFULLY.")
    print("=================================================================")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Radar-Return 3D Voxelisation Baseline")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to evaluate (default: 0)")
    parser.add_argument("--scene-idx", type=int, default=None, help="Scene index to evaluate (default: None)")
    parser.add_argument("--voxel-size", type=float, default=0.5, help="Voxel size in meters (default: 0.5)")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display interactive figure window")

    args = parser.parse_args()

    run_voxel_baseline(
        sample_idx=args.sample_idx,
        scene_idx=args.scene_idx,
        voxel_size=(args.voxel_size, args.voxel_size, args.voxel_size),
        output_dir=args.output_dir,
        show_plot=args.show
    )
