df1 = spark.read.csv("/Volumes/workspace/default/datajason/001_20250801_221250-3-2min.csv", header=True, inferSchema=True)
df2 = spark.read.csv("/Volumes/workspace/default/datajason/002_20250802_044544-3-2min.csv", header=True, inferSchema=True)
df3 = spark.read.csv("/Volumes/workspace/default/datajason/003_20250905_184138-3-2min.csv", header=True, inferSchema=True)

display(df1)
display(df2)
display(df3)

#------------------------------

#df1 

#DataFrame[timestamp: double, frame_number: int, nose_x: double, nose_y: double, left_eye_inner_x: double, left_eye_inner_y: double, left_eye_x: double, left_eye_y: double, left_eye_outer_x: double, left_eye_outer_y: double, right_eye_inner_x: double, right_eye_inner_y: double, right_eye_x: double, right_eye_y: double, right_eye_outer_x: double, right_eye_outer_y: double, left_ear_x: double, left_ear_y: double, right_ear_x: double, right_ear_y: double, mouth_left_x: double, mouth_left_y: double, mouth_right_x: double, mouth_right_y: double, left_shoulder_x: double, left_shoulder_y: double, right_shoulder_x: double, right_shoulder_y: double, left_elbow_x: double, left_elbow_y: double, right_elbow_x: double, right_elbow_y: double, left_wrist_x: double, left_wrist_y: double, right_wrist_x: double, right_wrist_y: double, left_pinky_x: double, left_pinky_y: double, right_pinky_x: double, right_pinky_y: double, left_index_x: double, left_index_y: double, right_index_x: double, right_index_y: double, left_thumb_x: double, left_thumb_y: double, right_thumb_x: double, right_thumb_y: double, left_hip_x: double, left_hip_y: double, right_hip_x: double, right_hip_y: double, left_knee_x: double, left_knee_y: double, right_knee_x: double, right_knee_y: double, left_ankle_x: double, left_ankle_y: double, right_ankle_x: double, right_ankle_y: double, left_heel_x: double, left_heel_y: double, right_heel_x: double, right_heel_y: double, left_foot_index_x: double, left_foot_index_y: double, right_foot_index_x: double, right_foot_index_y: double, nose_z: double, left_eye_inner_z: double, left_eye_z: double, left_eye_outer_z: double, right_eye_inner_z: double, right_eye_z: double, right_eye_outer_z: double, left_ear_z: double, right_ear_z: double, mouth_left_z: double, mouth_right_z: double, left_shoulder_z: double, right_shoulder_z: double, left_elbow_z: double, right_elbow_z: double, left_wrist_z: double, right_wrist_z: double, left_pinky_z: double, right_pinky_z: double, left_index_z: double, right_index_z: double, left_thumb_z: double, right_thumb_z: double, left_hip_z: double, right_hip_z: double, left_knee_z: double, right_knee_z: double, left_ankle_z: double, right_ankle_z: double, left_heel_z: double, right_heel_z: double, left_foot_index_z: double, right_foot_index_z: double, nose_confidence: double, left_eye_inner_confidence: double, left_eye_confidence: double, left_eye_outer_confidence: double, right_eye_inner_confidence: double, right_eye_confidence: double, right_eye_outer_confidence: double, left_ear_confidence: double, right_ear_confidence: double, mouth_left_confidence: double, mouth_right_confidence: double, left_shoulder_confidence: double, right_shoulder_confidence: double, left_elbow_confidence: double, right_elbow_confidence: double, left_wrist_confidence: double, right_wrist_confidence: double, left_pinky_confidence: double, right_pinky_confidence: double, left_index_confidence: double, right_index_confidence: double, left_thumb_confidence: double, right_thumb_confidence: double, left_hip_confidence: double, right_hip_confidence: double, left_knee_confidence: double, right_knee_confidence: double, left_ankle_confidence: double, right_ankle_confidence: double, left_heel_confidence: double, right_heel_confidence: double, left_foot_index_confidence: double, right_foot_index_confidence: double]

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Full list of 33 MediaPipe Pose keypoints (from your schema)
keypoints = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Add camera identifiers (as a new column) to each PySpark DF
df1 = df1.withColumn("camera", F.lit("cam1"))
df2 = df2.withColumn("camera", F.lit("cam2"))
df3 = df3.withColumn("camera", F.lit("cam3"))

# Stack the DataFrames vertically using PySpark union (extends rows; assumes identical schemas)
df_stacked = df1.union(df2).union(df3)

# Optional: Validate/align on frame_number (check for duplicates or missing sync)
df_stacked = df_stacked.orderBy("frame_number", "camera")
frame_counts = df_stacked.groupBy("frame_number").count().orderBy("frame_number")
frame_counts.show()  # Inspect: Should have ~3 rows per frame if all cameras present

# Convert to pandas for reshaping (efficient for this size)
df_pd = df_stacked.toPandas()

# Now proceed with pandas melting (group columns by type: x, y, z, confidence)
x_cols = [kp + '_x' for kp in keypoints]
y_cols = [kp + '_y' for kp in keypoints]
z_cols = [kp + '_z' for kp in keypoints]
conf_cols = [kp + '_confidence' for kp in keypoints]

# Melt each group to long format (stacks x/y/z/conf into rows)
df_x = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=x_cols, var_name='keypoint', value_name='x')
df_x['keypoint'] = df_x['keypoint'].str.replace('_x', '')

df_y = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=y_cols, var_name='keypoint', value_name='y')
df_y['keypoint'] = df_y['keypoint'].str.replace('_y', '')

df_z = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=z_cols, var_name='keypoint', value_name='z')
df_z['keypoint'] = df_z['keypoint'].str.replace('_z', '')

df_conf = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=conf_cols, var_name='keypoint', value_name='confidence')
df_conf['keypoint'] = df_conf['keypoint'].str.replace('_confidence', '')

# Merge the melted DFs on shared keys (creates long DF with up to 99 rows per frame)
df_long = (df_x.merge(df_y, on=['timestamp', 'frame_number', 'camera', 'keypoint'])
           .merge(df_z, on=['timestamp', 'frame_number', 'camera', 'keypoint'])
           .merge(df_conf, on=['timestamp', 'frame_number', 'camera', 'keypoint']))

# Sort for consistent ordering (frame -> camera -> keypoint)
df_long = df_long.sort_values(by=['frame_number', 'camera', 'keypoint']).reset_index(drop=True)

# Optional: Filter for sparsity (drop low-confidence points to make cloud sparser)
df_long = df_long[df_long['confidence'] >= 0.5].reset_index(drop=True)  # Adjust threshold

# Extract per-frame point clouds as numpy arrays (for PointNet++ Data objects)
point_clouds = []
for frame, group in df_long.groupby('frame_number'):
    # pos: [N, 3] where N <=99 (fewer if filtered)
    pos = group[['x', 'y', 'z']].values.astype(np.float32)
    
    # features: [N, 1] for confidence; expand as needed (e.g., add camera_id)
    features = group['confidence'].values[:, np.newaxis].astype(np.float32)
    
    # Add camera_id as a feature (e.g., 0 for cam1, 1 for cam2, 2 for cam3)
    camera_map = {'cam1': 0, 'cam2': 1, 'cam3': 2}
    camera_ids = pd.Series(group['camera'].map(camera_map)).values[:, np.newaxis].astype(np.float32)
    features = np.hstack([features, camera_ids])  # Now [N, 2]
    
    # Optional: Add keypoint_id as one-hot or index (33 classes)
    kp_ids = pd.factorize(group['keypoint'])[0][:, np.newaxis].astype(np.float32) / 32.0  # Normalize to [0,1]
    features = np.hstack([features, kp_ids])  # [N, 3]
    
    point_clouds.append((frame, pos, features))

# Example: Save or use for PyG
# print(f"Sample for frame {point_clouds[0][0]}: {len(point_clouds[0][1])} points")
# To create PyG Data: import torch; from torch_geometric.data import Data
# pos_t = torch.from_numpy(point_clouds[0][1]); x_t = torch.from_numpy(point_clouds[0][2])
# data = Data(pos=pos_t, x=x_t)

# For stride/subsampling (e.g., every 3rd frame for temporal efficiency)
stride = 3
strided_clouds = point_clouds[::stride]




#--------------------------

# Check the first few point clouds
print(f"Total frames processed: {len(point_clouds)}")
for i in range(min(3, len(point_clouds))):
    frame, pos, features = point_clouds[i]
    print(f"Frame {frame}: {pos.shape[0]} points, pos shape: {pos.shape}, features shape: {features.shape}")
    print(f"Sample pos (first 3 points):\n{pos[:3]}")
    print(f"Sample features (first 3 points):\n{features[:3]}")
    print(f"Min/Max confidence: {features[:, 0].min():.2f} / {features[:, 0].max():.2f}")
    print("-" * 40)

# Check for issues (e.g., NaNs or infinite values)
all_pos = np.vstack([p[1] for p in point_clouds])
all_feat = np.vstack([p[2] for p in point_clouds])
print(f"Overall: NaNs in pos? {np.isnan(all_pos).any()}")
print(f"Inf in pos? {np.isinf(all_pos).any()}")
print(f"Avg points per frame: {all_pos.shape[0] / len(point_clouds):.1f}")



#------------------------------------

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# MediaPipe Pose skeleton edges (example; full list from docs: nose-eyes-ears-shoulders-etc.)
# Indices 0=nose, 1=left_eye_inner, ..., 32=right_foot_index
skeleton_edges = torch.tensor([
    [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],  # Face connections (adjust to your indexing)
    [5, 6], [5, 7], [6, 8], [11, 12], [12, 14], [14, 16], [16, 18], [18, 20], [20, 22], [11, 13],
    [13, 15], [15, 17], [17, 19], [19, 21], [21, 23], [11, 24], [24, 26], [26, 28], [28, 30], [30, 32],
    # Add more; full: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
], dtype=torch.long).t()  # Shape [2, num_edges]

# Duplicate for 3 cameras: cam1 (0-32), cam2 (33-65), cam3 (66-98)
edge_index = []
for cam_offset in [0, 33, 66]:
    cam_edges = skeleton_edges + cam_offset
    edge_index.append(cam_edges)
edge_index = torch.cat(edge_index, dim=1)  # Combined [2, ~96 edges]

# Add self-loops for PointNetConv
edge_index = add_self_loops(edge_index)[0]

# Create Data list (one per frame)
data_list = []
for frame, pos_np, feat_np in point_clouds:
    pos = torch.from_numpy(pos_np).float()
    x = torch.from_numpy(feat_np).float()  # Node features (confidence, camera, kp_id)
    
    # Labels (y): Assume you have exercise labels per frame; otherwise, add later
    # For now, dummy: y = torch.tensor([0])  # e.g., class 0
    y = torch.tensor([frame % 2])  # Placeholder; replace with real labels
    
    data = Data(pos=pos, x=x, edge_index=edge_index, y=y)
    data_list.append(data)

print(f"Created {len(data_list)} PyG Data objects. Sample: {data_list[0]}")

#---
Created 28801 PyG Data objects. Sample: Data(x=[15, 3], edge_index=[2, 177], y=[1], pos=[15, 3])
#---


#--------------------------------

from torch_geometric.data import Dataset, DataLoader
from torch_geometric.transforms import Compose, NormalizeScale, RandomJitter, RandomRotate
import torch.nn.functional as F

class KettlebellDataset(Dataset):
    def __init__(self, data_list, transform=None, is_train=True):
        self.data_list = data_list
        self.transform = transform or Compose([
            NormalizeScale(),  # Unit sphere
            RandomJitter(0.01) if is_train else None,  # Noise for training
            RandomRotate(15) if is_train else None,  # Rotation
        ])
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].clone()  # Avoid modifying original
        if self.transform:
            data = self.transform(data)
        return data

# Usage
train_transform = Compose([NormalizeScale(), RandomJitter(0.01), RandomRotate(15)])
test_transform = NormalizeScale()

# Assume you split data_list (see previous train-test split advice)
# train_ds = KettlebellDataset(train_data_list, transform=train_transform, is_train=True)
# test_ds = KettlebellDataset(test_data_list, transform=test_transform, is_train=False)

# Batch loader
# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)


#-----------------------------

import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, knn_graph, global_max_pool

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):  # in_channels for x (features)
        super().__init__()
        self.sa1 = PointNetConv(nn.Sequential(nn.Linear(in_channels + 3, 64), nn.ReLU(), nn.Linear(64, 128)), add_self_loops=False)
        self.sa2 = PointNetConv(nn.Sequential(nn.Linear(128 + 3, 128), nn.ReLU(), nn.Linear(128, 256)), add_self_loops=False)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, pos, batch):
        # Sa1: Sample and group (use FPS + KNN for hierarchy)
        idx = fps(pos, batch, ratio=0.5)  # Sample to ~50 points
        row, col = knn_graph(pos, k=16, batch_x=batch, batch_y=batch[idx])  # Local neighborhoods
        edge_index = torch.stack([row, col], dim=0)
        x1 = self.sa1(x, pos, edge_index)
        
        # Sa2: On subsampled
        pos1 = pos[idx]
        idx2 = fps(pos1, batch[idx], ratio=0.5)
        row2, col2 = knn_graph(pos1, k=8, batch_x=batch[idx], batch_y=batch[idx2])
        edge_index2 = torch.stack([row2, col2], dim=0)
        x2 = self.sa2(x1[idx], pos1, edge_index2)
        
        # Global pooling and classify
        x_global = global_max_pool(x2, batch[idx][idx2])
        out = self.fc(x_global)
        return F.log_softmax(out, dim=1)

# Training loop (pseudo)
model = PointNetPlusPlus(num_classes=2)  # e.g., swing vs. other
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# For loader in training:
# for batch in train_loader:
#     optimizer.zero_grad()
#     out = model(batch.x, batch.pos, batch.batch)
#     loss = F.nll_loss(out, batch.y)
#     loss.backward()
#     optimizer.step()

#-------------------------------------

from pyspark.sql import functions as F

# Add camera column (string)
df1 = df1.withColumn("camera", F.lit("cam1"))
df2 = df2.withColumn("camera", F.lit("cam2"))
df3 = df3.withColumn("camera", F.lit("cam3"))

# Union (stack rows; assumes identical schemas)
df_stacked = df1.union(df2).union(df3)

# Order and validate sync (3 rows per frame ideally)
df_stacked = df_stacked.orderBy("frame_number", "camera")
frame_counts = df_stacked.groupBy("frame_number").count().orderBy("frame_number")
frame_counts.show(10)  # Check: count ~3 per frame; if not, join instead of union

# Convert to pandas for melt (efficient for keypoint reshaping; ok for <1M rows)
df_pd = df_stacked.toPandas()
print(f"Stacked shape: {df_pd.shape}")

#--------------------------------

import pandas as pd
import numpy as np

# 33 keypoints list (full from schema)
keypoints = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Melt groups
x_cols = [kp + '_x' for kp in keypoints]
y_cols = [kp + '_y' for kp in keypoints]
z_cols = [kp + '_z' for kp in keypoints]
conf_cols = [kp + '_confidence' for kp in keypoints]

df_x = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=x_cols, var_name='keypoint', value_name='x')
df_x['keypoint'] = df_x['keypoint'].str.replace('_x', '')

df_y = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=y_cols, var_name='keypoint', value_name='y')
df_y['keypoint'] = df_y['keypoint'].str.replace('_y', '')

df_z = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=z_cols, var_name='keypoint', value_name='z')
df_z['keypoint'] = df_z['keypoint'].str.replace('_z', '')

df_conf = pd.melt(df_pd, id_vars=['timestamp', 'frame_number', 'camera'], value_vars=conf_cols, var_name='keypoint', value_name='confidence')
df_conf['keypoint'] = df_conf['keypoint'].str.replace('_confidence', '')

# Merge to long DF: Now ~99 rows/frame (keypoint + camera + frame)
df_long = (df_x.merge(df_y, on=['timestamp', 'frame_number', 'camera', 'keypoint'])
           .merge(df_z, on=['timestamp', 'frame_number', 'camera', 'keypoint'])
           .merge(df_conf, on=['timestamp', 'frame_number', 'camera', 'keypoint']))

# Sort for consistent order (frame -> camera -> keypoint)
df_long = df_long.sort_values(by=['frame_number', 'camera', 'keypoint']).reset_index(drop=True)

print(f"Long DF shape: {df_long.shape}")
df_long.head()



timestamp	frame_number	camera	keypoint	x	y	z	confidence
0	  0.0	          0	      cam1	left_ankle	166.981137	268.503590	0.157004	0.111661
1	  0.0	          0	      cam1	left_ear	-13.184069	715.014191	-0.048632	0.999833
2	  0.0	          0	      cam1	left_elbow	164.760772	638.013420	-0.055798	0.140993
3	  0.0	          0	      cam1	left_eye	25.410264	761.150742	-0.077198	0.999865
4	  0.0	          0	      cam1	left_eye_inner	26.734870	759.556580	-0.087601	0.999901


#-------------------

# Filter (sparsity: drop low conf; adjust 0.5 -> 0.3 if too sparse)
df_long = df_long[df_long['confidence'] >= 0.5].reset_index(drop=True)

# Extract arrays per frame
point_clouds = []
for frame, group in df_long.groupby('frame_number'):
    pos = group[['x', 'y', 'z']].values.astype(np.float32)  # [N,3]
    features = group['confidence'].values[:, np.newaxis].astype(np.float32)  # [N,1]
    
    # Add camera_id (0-2)
    camera_map = {'cam1': 0, 'cam2': 1, 'cam3': 2}
    camera_ids = group['camera'].map(camera_map).values[:, np.newaxis].astype(np.float32)
    features = np.hstack([features, camera_ids])  # [N,2]
    
    # Add kp_id (0-32 normalized)
    kp_ids = pd.factorize(group['keypoint'])[0][:, np.newaxis].astype(np.float32) / 32.0
    features = np.hstack([features, kp_ids])  # [N,3]
    
    point_clouds.append((frame, pos, features))

print(f"Total frames/point_clouds: {len(point_clouds)}")
print(f"Avg points/frame: {np.mean([pc[1].shape[0] for pc in point_clouds]):.1f}")


#---
Total frames/point_clouds: 28801
Avg points/frame: 68.5
#---


#----------------------------

# Inspect first 3
for i in range(min(3, len(point_clouds))):
    frame, pos, features = point_clouds[i]
    print(f"Frame {frame}: {pos.shape[0]} points, pos:\n{pos[:3]}, features:\n{features[:3]}")
    print(f"Conf min/max: {features[:,0].min():.2f}/{features[:,0].max():.2f}\n")

# Global check
all_pos = np.vstack([pc[1] for pc in point_clouds])
print(f"NaNs: {np.isnan(all_pos).any()}, Inf: {np.isinf(all_pos).any()}")


#---
Frame 0: 15 points, pos:
[[-1.3184069e+01  7.1501416e+02 -4.8632383e-02]
 [ 2.5410263e+01  7.6115076e+02 -7.7198386e-02]
 [ 2.6734871e+01  7.5955658e+02 -8.7601423e-02]], features:
[[0.99983287 0.         0.        ]
 [0.9998652  0.         0.03125   ]
 [0.99990094 0.         0.0625    ]]
Conf min/max: 1.00/1.00

Frame 1: 30 points, pos:
[[ 5.1674037e+00  7.2638562e+02 -6.6115521e-02]
 [ 4.7168686e+01  7.8114618e+02 -7.7220298e-02]
 [ 4.8135509e+01  7.7903931e+02 -8.7633938e-02]], features:
[[0.9998329  0.         0.        ]
 [0.99986213 0.         0.03125   ]
 [0.99989796 0.         0.0625    ]]
Conf min/max: 0.87/1.00

Frame 2: 30 points, pos:
[[ 5.60168982e+00  7.31123657e+02 -9.98427942e-02]
 [ 4.86870079e+01  7.88149719e+02 -1.00635529e-01]
 [ 4.95432663e+01  7.85999268e+02 -1.11728944e-01]], features:
[[0.99982697 0.         0.        ]
 [0.99985206 0.         0.03125   ]
 [0.99989027 0.         0.0625    ]]
Conf min/max: 0.89/1.00

NaNs: False, Inf: False
#-----------------

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, subgraph

# Full skeleton (32 edges for 33 points)
skeleton_edges = torch.tensor([
    [0,1],[1,3],[3,5],[0,2],[2,4],[4,6],[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],
    [11,12],[11,13],[13,15],[15,17],[17,19],[19,21],[12,14],[14,16],[16,18],[18,20],[20,22],
    [11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]
], dtype=torch.long).t().contiguous()

data_list = []
min_points = 10
N_full = 99

for frame, pos_np, feat_np in point_clouds:
    pos_full = torch.from_numpy(pos_np).float()
    x_full = torch.from_numpy(feat_np).float()
    
    actual_n = feat_np.shape[0]
    valid_mask = torch.ones(N_full, dtype=torch.bool)
    
    if actual_n > 0:
        conf_torch = torch.from_numpy(feat_np[:, 0])
        conf_mask = conf_torch >= 0.5
        valid_mask[:actual_n] = conf_mask
        valid_mask[actual_n:] = False
    
    num_valid = valid_mask.sum().item()
    if num_valid < min_points:
        continue
    
    # Full edges
    full_edge_index = []
    for offset in [0, 33, 66]:
        cam_edges = skeleton_edges.clone() + offset
        full_edge_index.append(cam_edges)
    full_edge_index = torch.cat(full_edge_index, dim=1)
    
    # Filter/remap
    edge_index_filtered, subset = subgraph(valid_mask, full_edge_index, num_nodes=N_full)
    pos = pos_full[subset]
    x = x_full[subset]
    
    edge_index = add_self_loops(edge_index_filtered)[0]
    y = torch.tensor([frame % 3], dtype=torch.long)  # Placeholder
    
    data = Data(pos=pos, x=x, edge_index=edge_index, y=y)
    data_list.append(data)

print(f"Created {len(data_list)} Data objects. Avg nodes: {sum(d.num_nodes for d in data_list)/len(data_list):.1f}")

#---
Created 28801 Data objects. Avg nodes: 1.0
#---

#------------------------------

print(data_list[0])

#---
Data(x=[1, 15, 3], edge_index=[2, 31], y=[1], pos=[1, 15, 3])
#---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json
from torch_geometric.utils import to_networkx
import networkx as nx
from IPython.display import HTML
import base64
import io
import torch

volume_path = "/Volumes/main/default/jason_results/"  # Your new "FileStore"

# a. Data Verification (As Before)
print("=== Data Verification ===")
print(f"Total Data objects: {len(data_list)}")
points_per_frame = [d.num_nodes for d in data_list]
print(f"Avg points/frame: {np.mean(points_per_frame):.1f}")
print(f"Min/Max points: {min(points_per_frame)} / {max(points_per_frame)}")

sample_data = data_list[0]
print(f"\nSample Frame 0: pos shape {sample_data.pos.shape}, x shape {sample_data.x.shape}")
print(f"Edge max index: {sample_data.edge_index.max().item()} < {sample_data.num_nodes}")
print("Sample pos (first 3):")
print(sample_data.pos[:3].numpy())

if 'df_long' in locals():
    df_long_sample = df_long.head(5)[['frame_number', 'camera', 'keypoint', 'x', 'y', 'z', 'confidence']]
else:
    df_long_sample = pd.DataFrame()
print(f"\nLong DF Sample (melt/stack):")
print(df_long_sample)

# b. Model Verification (Safe)
print("\n=== Model Verification ===")
if 'model' in locals() and 'test_preds' in locals() and 'test_trues' in locals():
    test_trues_flat = np.array(test_trues).flatten()
    test_preds_flat = np.array(test_preds).flatten()
    cm = confusion_matrix(test_trues_flat, test_preds_flat)
    cm_df = pd.DataFrame(cm, index=['Class 0', 'Class 1', 'Class 2'], columns=['Class 0', 'Class 1', 'Class 2'])
    
    # Save CM CSV via dbutils
    cm_csv = cm_df.to_csv(index=True)
    dbutils.fs.put(volume_path + "confusion_matrix.csv", cm_csv, overwrite=True)
    print("Confusion Matrix:\n", cm_df)
    
    report = classification_report(test_trues_flat, test_preds_flat, output_dict=True)
    report_df = pd.DataFrame(report).round(3)
    report_csv = report_df.to_csv(index=True)
    dbutils.fs.put(volume_path + "classification_report.csv", report_csv, overwrite=True)
    print("\nClassification Report:\n", report_df)
    
    summary = {
        "total_samples": len(data_list),
        "avg_points_per_frame": np.mean(points_per_frame),
        "test_accuracy": np.mean(test_preds_flat == test_trues_flat),
        "best_val_acc": best_val_acc if 'best_val_acc' in locals() else "N/A"
    }
    summary_str = json.dumps(summary, indent=4)
    dbutils.fs.put(volume_path + "training_summary.json", summary_str, overwrite=True)
    print("\nTraining Summary:", summary)
else:
    summary = {"total_samples": len(data_list), "avg_points": np.mean(points_per_frame), "data_verified": True}
    summary_str = json.dumps(summary, indent=4)
    dbutils.fs.put(volume_path + "training_summary.json", summary_str, overwrite=True)
    print("Basic Summary saved via dbutils")

# c. Model Save (Direct to Volume)
if 'model' in locals():
    torch.save(model.state_dict(), volume_path + "pointnetpp_model.pth")
    print("Model saved to Volume")

# Sample Data (Direct)
torch.save(data_list[:100], volume_path + "sample_data.pt")
print("Sample data saved to Volume")

# d. Plot (Base64 Fallback)
def plot_sample_graph_base64(data, volume_path_save=volume_path + "sample_pointcloud.png"):
    G = to_networkx(data, to_undirected=True)
    pos_2d = data.pos.squeeze()[:, :2].numpy() if data.pos.dim() > 2 else data.pos[:, :2].numpy()
    plt.figure(figsize=(10,8))
    nx.draw(G, pos_2d, node_color=data.x.squeeze()[:,0].numpy(), cmap=plt.cm.Blues, node_size=100, with_labels=False, edge_color='gray')
    plt.title("Sample Point Cloud (Keypoints as Nodes, Skeleton Edges, Color=Confidence; ~15/99 Sparse)")
    
    try:
        plt.savefig(volume_path_save, dpi=150, bbox_inches='tight')
        print(f"Plot saved directly to {volume_path_save}")
    except OSError as e:
        print(f"Direct save failed: {e}. Using base64 download.")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        html = f'<a href="data:image/png;base64,{b64}" download="sample_pointcloud.png">Download Plot PNG</a>'
        display(HTML(html))
    
    plt.show()
    plt.close()

plot_sample_graph_base64(sample_data)

# e. Stride Sample
if len(point_clouds) > 3:
    stride_sample = point_clouds[::3][:3]
    print(f"\nStride Sample (every 3rd frame): {len(stride_sample)} frames, avg points {np.mean([s[1].shape[0] for s in stride_sample]):.1f}")

print("\n=== Verification Complete: Saved to Volume (No DBFS Error). Download via URL/CLI! ===")

#---------

import pandas as pd

# Assume df_long exists; if not, regenerate from earlier steps
# Filter for a specific frame (e.g., frame 0; change to your frame of interest)
frame_num = 0  # Or df_long['frame_number'].iloc[0]
frame_data = df_long[df_long['frame_number'] == frame_num]

# Pivot to wide: Cameras as columns for x/y/z/conf (side-by-side)
pivot_x = frame_data.pivot(index='keypoint', columns='camera', values='x').fillna('N/A')
pivot_y = frame_data.pivot(index='keypoint', columns='camera', values='y').fillna('N/A')
pivot_z = frame_data.pivot(index='keypoint', columns='camera', values='z').fillna('N/A')
pivot_conf = frame_data.pivot(index='keypoint', columns='camera', values='confidence').fillna('N/A')

print(f"Side-by-Side for Frame {frame_num} (3 Cameras Next to Each Other per Keypoint):")
print("X Values (cam1 | cam2 | cam3):\n", pivot_x)
print("\nY Values:\n", pivot_y)
print("\nZ Values:\n", pivot_z)
print("\nConfidence:\n", pivot_conf)

# Combined Multi-Index DF (Full Wide Matrix View)
combined = pd.concat([pivot_x, pivot_y, pivot_z, pivot_conf], keys=['x', 'y', 'z', 'conf'], axis=1)
print("\nCombined Wide Matrix (Full 33x12 View; 3 Cams Side-by-Side for Each Coord):\n", combined.head(10))  # Head for top 10 kps

# Number of keypoints/cameras in this frame (pre-filter)
print(f"\nPoints in Frame {frame_num}: {len(frame_data)} (full ~99; low if filtered)")

#--
Side-by-Side for Frame 0 (3 Cameras Next to Each Other per Keypoint):
X Values (cam1 | cam2 | cam3):
 camera                cam1
keypoint                  
left_ear        -13.184069
left_eye         25.410264
left_eye_inner   26.734870
left_eye_outer   25.555704
left_hip         17.675705
left_shoulder    48.890335
mouth_left       20.830927
mouth_right      30.420807
nose             37.322233
right_ear        31.524319
right_eye        29.394361
right_eye_inner  28.306564
right_eye_outer  25.261863
right_hip       -19.642156
right_shoulder   22.178560

Y Values:
 camera                 cam1
keypoint                   
left_ear         715.014191
left_eye         761.150742
left_eye_inner   759.556580
left_eye_outer   760.553055
left_hip           9.015143
left_shoulder    583.985558
mouth_left       732.838593
mouth_right      688.728638
nose             753.526154
right_ear        659.842453
right_eye        773.520660
right_eye_inner  779.540100
right_eye_outer  757.415771
right_hip         -3.949845
right_shoulder   587.936401

Z Values:
 camera               cam1
keypoint                 
left_ear        -0.048632
left_eye        -0.077198
left_eye_inner  -0.087601
left_eye_outer  -0.077123
left_hip        -0.059700
left_shoulder   -0.049261
mouth_left      -0.059389
mouth_right     -0.067027
nose            -0.083399
right_ear        0.011715
right_eye       -0.097250
right_eye_inner -0.087626
right_eye_outer -0.077184
right_hip        0.060910
right_shoulder   0.207782

Confidence:
 camera               cam1
keypoint                 
left_ear         0.999833
left_eye         0.999865
left_eye_inner   0.999901
left_eye_outer   0.999894
left_hip         0.999249
left_shoulder    0.999327
mouth_left       0.999143
mouth_right      0.999307
nose             0.999931
right_ear        0.999829
right_eye        0.999934
right_eye_inner  0.999946
right_eye_outer  0.999956
right_hip        0.999517
right_shoulder   0.998414

Combined Wide Matrix (Full 33x12 View; 3 Cams Side-by-Side for Each Coord):
                         x           y         z      conf
camera               cam1        cam1      cam1      cam1
keypoint                                                 
left_ear       -13.184069  715.014191 -0.048632  0.999833
left_eye        25.410264  761.150742 -0.077198  0.999865
left_eye_inner  26.734870  759.556580 -0.087601  0.999901
left_eye_outer  25.555704  760.553055 -0.077123  0.999894
left_hip        17.675705    9.015143 -0.059700  0.999249
left_shoulder   48.890335  583.985558 -0.049261  0.999327
mouth_left      20.830927  732.838593 -0.059389  0.999143
mouth_right     30.420807  688.728638 -0.067027  0.999307
nose            37.322233  753.526154 -0.083399  0.999931
right_ear       31.524319  659.842453  0.011715  0.999829

Points in Frame 0: 15 (full ~99; low if filtered)
#---
