import numpy as np
import open3d as o3d

col_pcd = o3d.io.read_point_cloud("/home/otter77/3d_model/colmap.ply")
gl_pcd = o3d.io.read_point_cloud("/home/otter77/3d_model/gt_gl.ply")

print("Apply point-to-point ICP")
threshold = 1000
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], 
                         [0.0, 0.0, 0.0, 1.0]])
reg_p2p = o3d.pipelines.registration.registration_icp(
    col_pcd, gl_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

with open("output.txt", "a") as file:
    file.write("colmap:\n")
    file.write(f"{reg_p2p}\n")
    file.write("Transformation is: \n")
    file.write(f"{reg_p2p.transformation}\n")

col_pcd.transform(reg_p2p.transformation)
o3d.io.write_point_cloud("/home/otter77/3d_model/col_trans.ply", col_pcd)