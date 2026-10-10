import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

# 1. Initialize dataset map
print("🔄 Loading dataset inputs for scene mapping...")
nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)

# Target the busy intersection scene with the bus (Index 4)
SCENE_INDEX = 4  
scene = nusc.scene[SCENE_INDEX]
print(f"🎬 Processing Target Scene: {scene['name']} - {scene['description']}")

# 2. Configure video saver parameters
frame_width, frame_height = 1600, 900
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('oncoming_hidden_traffic.mp4', fourcc, 10, (frame_width, frame_height))

current_sample_token = scene['first_sample_token']

print("🚨 Extracting and tracing non-line-of-sight targets...")
while current_sample_token != '':
    sample = nusc.get('sample', current_sample_token)
    img_path = nusc.get_sample_data_path(sample['data']['CAM_FRONT'])
    radar_path = nusc.get_sample_data_path(sample['data']['RADAR_FRONT'])
    
    frame = cv2.imread(img_path)
    pc = RadarPointCloud.from_file(radar_path)
    radar_points = pc.points # Attributes: [X, Y, Z, RCS, Vx, Vy, ...]
    
    # 3. DEFINE OCCLUSION AND BOUNDARY LOGIC
    # Assume a large vehicle (like a bus/truck) is blocking the view starting at 10 meters ahead.
    # Anything beyond 10m inside the center-left column is visually occluded.
    blocker_end_y = 10.0      
    camera_blind_left = -4.0  # Widen slightly to cover oncoming lanes/bus width
    camera_blind_right = 4.0
    
    for i in range(radar_points.shape[1]):
        x = radar_points[0, i]   # Lateral distance (Left/Right)
        y = radar_points[1, i]   # Longitudinal depth distance (Ahead)
        vy = radar_points[7, i]  # Relative Doppler Speed along Y-axis
        
        # CONDITION A: Is the radar echo deeper than our trailing vehicle block?
        if y > blocker_end_y and camera_blind_left <= x <= camera_blind_right:
            
            # CONDITION B: Is it moving towards us? (Negative velocity indicates approaching target)
            if vy < -0.5:
                
                # Project the raw 3D spatial coordinate straight into 2D camera pixels
                points_jacobian, depth = nusc.explorer.map_pointcloud_to_image(
                    sample['data']['CAM_FRONT'], 
                    pointsensor_token=sample['data']['RADAR_FRONT']
                )
                
                pixel_x = int(points_jacobian[0, i])
                pixel_y = int(points_jacobian[1, i])
                
                # Render the indicator if it maps onto our screen matrix
                if 0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height:
                    # Draw a distinct Orange alert box for hidden oncoming elements
                    cv2.rectangle(frame, (pixel_x - 45, pixel_y - 75), (pixel_x + 45, pixel_y + 15), (0, 140, 255), 3)
                    cv2.putText(frame, f"HIDDEN TARGET: {abs(vy):.1f} m/s", (pixel_x - 90, pixel_y - 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
                    
                    # Draw a red warning dot at the anchor base
                    cv2.circle(frame, (pixel_x, pixel_y), 6, (0, 0, 255), -1)
    
    video_writer.write(frame)
    current_sample_token = sample['next']

video_writer.release()
print("🎉 Success! Your video 'oncoming_hidden_traffic.mp4' has been compiled.")
