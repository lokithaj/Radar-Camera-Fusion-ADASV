from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='.', verbose=False)

print("🔍 Searching for scenes with oncoming traffic or heavy occlusions...")
for i, scene in enumerate(nusc.scene):
    desc = scene['description'].lower()
    # Looking for scenes with active oncoming cars or big buses/trucks blocking views
    if "oncoming" in desc or "bus" in desc or "heavy traffic" in desc:
        print(f"Index: {i} | Name: {scene['name']} | Description: {desc}")
