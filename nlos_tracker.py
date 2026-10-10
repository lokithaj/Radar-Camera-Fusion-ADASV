import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

print("🔄 Loading dataset inputs...")
nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)
scene = nusc.scene[0] 
print(f"🎬 Processing Scene: {scene['name']} ({scene['description']})")

frame_width, frame_height = 1600, 900
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('nlos_ghost_tracking_demo.mp4', fourcc, 10, (frame_width, frame_height))

current_sample_token = scene['first_sample_token']

print("🔍 Analyzing frames for visual occlusion...")
while current_sample_token != '':
    sample = nusc.get('sample', current_sample_token)
    img_path = nusc.get_sample_data_path(sample['data']['CAM_FRONT'])
    radar_path = nusc.get_sample_data_path(sample['data']['RADAR_FRONT'])
    
    frame = cv2.imread(img_path)
    pc = RadarPointCloud.from_file(radar_path)
    radar_points = pc.points
    
    # Custom Non-Line-of-Sight (NLOS) Tracking Boundaries
    blocker_distance_y = 12.0  
    blind_zone_left = -2.0     
    blind_zone_right = 2.0     
    
    for i in range(radar_points.shape[1]):
        x = radar_points[0, i]   
        y = radar_points[1, i]   
        vx = radar_points[6, i]  
        
        if y > blocker_distance_y and blind_zone_left <= x <= blind_zone_right:
            if abs(vx) > 0.3:  
                points_jacobian, depth = nusc.explorer.map_pointcloud_to_image(
                    sample['data']['CAM_FRONT'], 
                    pointsensor_token=sample['data']['RADAR_FRONT']
                )
                pixel_x = int(points_jacobian[0, i])
                pixel_y = int(points_jacobian[1, i])
                
                if 0 <= pixel_x < frame_width and 0 <= pixel_y < frame_height:
                    cv2.rectangle(frame, (pixel_x - 50, pixel_y - 80), (pixel_x + 50, pixel_y + 10), (255, 191, 0), 3)
                    cv2.putText(frame, "GHOST TARGET (OCCLUDED)", (pixel_x - 70, pixel_y - 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 191, 0), 2)
    
    video_writer.write(frame)
    current_sample_token = sample['next']

video_writer.release()
print("🎉 Success! Your video 'nlos_ghost_tracking_demo.mp4' has been created and saved.")
