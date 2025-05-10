from pytorch3d.loss import chamfer_distance
import torch
import numpy as np
import open3d as o3d

col_pcd = o3d.io.read_point_cloud("/home/otter77/3d_model/ace0_trans.ply")
gt_pcd = o3d.io.read_point_cloud("/home/otter77/3d_model/gt_gl.ply")

# 提取点云坐标，并转换为 NumPy 数组
source_points = np.asarray(col_pcd.points)
target_points = np.asarray(gt_pcd.points)

print("CUDA is", torch.cuda.is_available())

# 转换 NumPy 数组为 PyTorch Tensor
source_tensor = torch.tensor(source_points, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
target_tensor = torch.tensor(target_points, dtype=torch.float32).unsqueeze(0).cuda()  #

# By default: average over points then batches
loss_chamfer, _ = chamfer_distance(
    source_tensor, target_tensor,
    batch_reduction="mean",    # or "sum" / "none"
    point_reduction="mean"     # or "sum" / "none"
)

with open("output.txt", "a") as file:
    file.write(f"ACE0 Chamfer loss: {loss_chamfer.item():.6f}\n")