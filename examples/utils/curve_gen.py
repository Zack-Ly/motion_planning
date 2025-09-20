# curve_gen.py
import numpy as np

def gen_circle(center, radius=0.05, num_points=100, z=0.0, quat=[0,0,0,1]):
    """
    生成 XY 平面上的圆轨迹
    - center: (x, y, z) 圆心
    - radius: 半径
    - num_points: 轨迹点数
    - z: 固定的 z 坐标 (默认用 center[2])
    - quat: 末端姿态 (保持不变)

    返回 [(pos, quat, duration), ...]
    """
    if z == 0.0:
        z = center[2]

    traj = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        pos = [x, y, z]
        traj.append((pos, quat, 0.05))  # 每步0.05秒，可改
    return traj
