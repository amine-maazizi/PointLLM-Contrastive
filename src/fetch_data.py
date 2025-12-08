import os
import random
import numpy as np
import objaverse
import trimesh

SAMPLE_SIZE = 20_000
POINTS_PER_CLOUD = 2000
DATA_DIR = "data"

# Load all UIDs and sample a subset
all_uids = objaverse.load_uids()
random.seed(42)  # reproducible
sample_uids = random.sample(all_uids, SAMPLE_SIZE)
annotations = objaverse.load_annotations(uids=sample_uids)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

saved_count = 0
for uid in sample_uids:
    try:
        # Load the mesh
        path = objaverse.load_objects(uids=[uid])[uid]
        mesh = trimesh.load(path, force='mesh')
        
        # Skip empty meshes
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            print(f"Skipping {uid}: Empty mesh")
            continue
        
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            print(f"Skipping {uid}: No faces")
            continue
        
        # Sample point cloud
        points = mesh.sample(POINTS_PER_CLOUD)
        
        # Prepare safe filename
        name = annotations[uid].get("name", uid)
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        dest_path = os.path.join(DATA_DIR, f"{safe_name}.npz")
        
        # Save point cloud as compressed numpy array
        np.savez_compressed(dest_path, points=points)
        saved_count += 1
        
        if saved_count % 100 == 0:
            print(f"Saved {saved_count} point clouds...")
            
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        continue
    
print(f"Saved {saved_count}/{len(sample_uids)} downsampled point clouds to '{DATA_DIR}'")
