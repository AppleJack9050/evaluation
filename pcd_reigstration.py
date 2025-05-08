import numpy as np
import open3d as o3d
from py3dtk.registration import NDT
from typing import Tuple

def register_ndt_py3dtk(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    resolution: float = 1.0,
    max_iter: int = 30
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    # Convert to numpy
    src_np = np.asarray(source.points, dtype=np.float64)
    tgt_np = np.asarray(target.points, dtype=np.float64)

    # Run NDT
    ndt = NDT(resolution=resolution, max_iter=max_iter)
    T, info = ndt.run(src_np, tgt_np)
    print(f"[NDT] info={info}")

    # Apply transform
    aligned = source.clone()
    aligned.transform(T)
    return aligned, T
