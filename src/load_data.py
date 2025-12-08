import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR = "data/"

def load_point_clouds():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
    labels = []
    point_clouds = []
    
    for file in files:
        path = os.path.join(DATA_DIR, file)
        data = np.load(path)
        points = data["points"]
        point_clouds.append(points)
        
        label = os.path.splitext(file)[0]
        labels.append(label)
    
    return point_clouds, labels

if __name__ == "__main__":
    size = 10
    rows = 2
    cols = 5

    fig = plt.figure(figsize=(20, 8))
    point_clouds, labels = load_point_clouds()

    for i, (points, label) in enumerate(zip(point_clouds[:size], labels[:size])):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c='b', depthshade=True)
        ax.set_title(label)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
