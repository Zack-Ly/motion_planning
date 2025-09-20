#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pybullet as p
import pybullet_data

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as RosDuration

import yaml  # pip install pyyaml 或 sudo apt install python3-yaml
import sys

from utils.curve_gen import gen_circle
# 在原始文件开头添加
import matplotlib.pyplot as plt
from spatialmath import SE3

def generate_circle_xy(center, radius, num_points=100, z=0.5):
    """
    生成 XY 平面圆的轨迹
    center: (x0,y0)
    radius: 半径
    z: 高度固定
    """
    x0, y0 = center
    thetas = np.linspace(0, 2*math.pi, num_points)
    traj = []
    for th in thetas:
        pos = [x0 + radius*math.cos(th), y0 + radius*math.sin(th), z]
        quat = [0, 1, 0, 0]   # 固定朝向，需根据实际调整
        traj.append((pos, quat))
    return traj

def norm_quat_xyzw(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return (q / (n if n > 0 else 1.0)).tolist()


class PyBulletIKVizNode(Node):
    """
    - PyBullet 加载 URDF、做 IK、可视化（GUI 常驻）
    - 订阅 /joint_states，把真实机器人状态镜像到 PyBullet
    - 多目标点：每个目标点 = IK -> 平滑动画 -> 通过 FollowJointTrajectory 发送到控制器
    """

    def __init__(self):
        super().__init__("pybullet_ik_viz_action")

        # ========== 可根据自己项目修改的参数 ==========
        self.urdf_path = "examples/urdf/cr5_robot.urdf"
        self.add_search_path = pybullet_data.getDataPath()  # 若有自定义 mesh 路径可再追加
        self.use_fixed_base = True

        # 末端 link：强烈建议用名字自动查找，避免手改索引
        self.ee_link_name = "tool0"     # 改成你的末端 link 名称
        self.fallback_ee_link_index = 7 # 如果名字查不到，就用这个索引

        # ROS2 控制器 action 名称（确认你的控制器名字）
        self.follow_traj_action = "/cr5_group_controller/follow_joint_trajectory"

        # 订阅的关节状态话题
        self.joint_states_topic = "/joint_states"

        # 可选：如果 ROS 里的关节名与 URDF 不一致，可在此做映射：ros_name -> urdf_name
        self.name_map_ros_to_urdf: Dict[str, str] = {
            # "ros_joint_1": "urdf_joint_1",
        }

        # IK 与动画配置
        self.ik_max_iters = 200
        self.ik_tol = 1e-5
        self.anim_steps = 200           # 平滑动画插值步数（越大越平滑）
        self.gui_rate_hz = 240          # PyBullet 刷新频率

        # 目标点序列（示例：两个目标）
        # 每个元素：(pos[x,y,z], quat[x,y,z,w], 到达时长sec)
        self.targets: List[Tuple[List[float], List[float], float]] = [
            ([0.472669, -0.14159, 0.468731], norm_quat_xyzw([-0.707, 0.707, 0.0, 0.0]), 3.0),
            ([0.40, -0.10, 0.55], norm_quat_xyzw([-0.707, 0.707, 0.0, 0.0]), 3.0),
        ]
        # ============================================

        # 存放参考轨迹与实际轨迹
        self.ref_traj = []
        self.real_traj = []

        # 允许通过参数指定 YAML 路径
        self.declare_parameter('targets_yaml', '')
        yaml_path = self.get_parameter('targets_yaml').get_parameter_value().string_value

        if yaml_path:
            loaded = self._load_targets_from_yaml(yaml_path)
            if loaded:
                self.targets = loaded
                self.get_logger().info(f"Loaded {len(self.targets)} targets from YAML: {yaml_path}")
            else:
                self.get_logger().warn(f"Failed to load targets from YAML: {yaml_path}; keep default self.targets")
                
        # ---- PyBullet 初始化 ----
        if p.isConnected():
            p.disconnect()
        p.connect(p.GUI)  # GUI 模式
        p.setAdditionalSearchPath(self.add_search_path)

        # 加载地面
        self.plane = p.loadURDF("plane.urdf", basePosition=[0,0,0], baseOrientation=[0,0,0,1], useFixedBase=True)

        # 再加载机械臂
        self.robot = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        # 记录 URDF 的关节与限位
        self.movable_joints: List[int] = []
        self.urdf_joint_names: List[str] = []
        self.lower_limits: List[float] = []
        self.upper_limits: List[float] = []
        self.joint_ranges: List[float] = []
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            jtype = info[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.movable_joints.append(j)
                name = info[1].decode("utf-8")
                self.urdf_joint_names.append(name)
                lo, hi = float(info[8]), float(info[9])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    # continuous 或未定义边界
                    lo, hi = -math.pi, math.pi
                self.lower_limits.append(lo)
                self.upper_limits.append(hi)
                self.joint_ranges.append(hi - lo)

        self.lower_limits = np.array(self.lower_limits)
        self.upper_limits = np.array(self.upper_limits)
        self.joint_ranges = np.array(self.joint_ranges)


        self.ee_link = 7

        self.get_logger().info(
            f"URDF 关节数={len(self.movable_joints)}; tip_link index={self.ee_link}; "
            f"joint_names={self.urdf_joint_names}"
        )

        # ---- ROS2 订阅 /joint_states：镜像到 PyBullet ----
        self.latest_js: Dict[str, float] = {}
        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 50)

        # ---- ROS2 Action 客户端 ----
        self.client = ActionClient(self, FollowJointTrajectory, self.follow_traj_action)

    # ---------- 工具 ----------
    def _find_link_index_by_name(self, link_name: str) -> Optional[int]:
        """根据 link 名称查找其在 PyBullet 的索引"""
        if not link_name:
            return None
        for j in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, j)[12].decode("utf-8") == link_name:
                return j
        return None

    def _joint_state_cb(self, msg: JointState):
        """缓存最新 /joint_states"""
        for name, pos in zip(msg.name, msg.position):
            urdf_name = self.name_map_ros_to_urdf.get(name, name)
            self.latest_js[urdf_name] = float(pos)

    def get_q_now(self) -> np.ndarray:
        """优先用 /joint_states 构建当前关节向量；否则用 PyBullet 的当前状态"""
        if self.latest_js:
            q = []
            for name, j in zip(self.urdf_joint_names, self.movable_joints):
                q.append(self.latest_js.get(name, p.getJointState(self.robot, j)[0]))
            return np.array(q, dtype=float)
        # 无 joint_states 时回退
        return np.array([p.getJointState(self.robot, j)[0] for j in self.movable_joints], dtype=float)
    
    def _load_targets_from_yaml(self, path):
        """
        读取 YAML:
        targets:
            - pos: [x,y,z]
            quat_xyzw: [x,y,z,w]
            duration: 3.0
        返回: List[ (pos, quat_xyzw, duration) ]
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Read YAML failed: {e}")
            return None

        items = data.get('targets', [])
        out = []
        for i, it in enumerate(items, 1):
            pos  = it.get('pos') or it.get('position')
            quat = it.get('quat_xyzw') or it.get('quat') or it.get('orientation')
            dur  = it.get('duration', 3.0)

            if not (isinstance(pos, list) and len(pos) == 3):
                self.get_logger().warn(f"targets[{i}] missing pos[3]; skip")
                continue
            if not (isinstance(quat, list) and len(quat) == 4):
                self.get_logger().warn(f"targets[{i}] missing quat_xyzw[4]; skip")
                continue
            try:
                dur = float(dur)
            except Exception:
                dur = 3.0

            out.append([list(map(float, pos)), norm_quat_xyzw(quat), dur])

        return out

    def follow_cartesian_traj(self, traj_points, duration=5.0):
        """
        traj_points: [(pos, quat), ...]
        duration: 总时长
        """
        q_now = self.get_q_now()
        n = len(traj_points)
        for i, (pos, quat) in enumerate(traj_points):
            # IK
            q_star = self.solve_ik(pos, quat, q_now)
            if q_star is None:
                self.get_logger().warn(f"点{i} IK 失败，跳过")
                continue

            # 动画到位（PyBullet）
            ok = self.animate_to_q(q_star, steps=5)
            if not ok:
                self.get_logger().error(f"点{i} 碰撞，退出")
                break

            # 执行控制器
            self.send_to_action(q_star, duration_sec=duration/n)

            # 记录参考轨迹
            self.ref_traj.append(pos)

            # 记录实际末端位置（用 PyBullet forward kinematics）
            link_state = p.getLinkState(self.robot, self.ee_link)
            self.real_traj.append(link_state[0])  # 取 position

            q_now = q_star

    # ---------- 可视化 ----------
    def mirror_joint_states_to_pybullet(self):
        """把最新 joint_states 直接同步到 PyBullet，可做‘镜像模式’可视化"""
        if not self.latest_js:
            return
        for name, j in zip(self.urdf_joint_names, self.movable_joints):
            pos = self.latest_js.get(name, None)
            if pos is not None:
                p.resetJointState(self.robot, j, pos)

    def animate_to_q(self, q_target, steps=100):
        q_now = self.get_q_now()
        for k in range(steps):
            q = q_now + (q_target - q_now) * (k + 1) / steps
            for j, val in zip(self.movable_joints, q):
                p.resetJointState(self.robot, j, val)

            # 检查地面碰撞（忽略 link1）
            contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.plane)
            bad = [c for c in contacts if c[3] != 1]
            if bad:
                self.get_logger().error(
                    f"动画第{k}步 link{bad[0][3]} 碰撞地面，中断，程序退出"
                )
                return False  # 返回失败

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        return True  # 正常完成



    # ---------- IK ----------
    def solve_ik(self, target_pos, target_quat_xyzw, q_now):
        target_quat = norm_quat_xyzw(target_quat_xyzw)
        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=int(self.ee_link),
            targetPosition=target_pos,
            targetOrientation=target_quat,
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=self.joint_ranges.tolist(),
            restPoses=q_now.tolist(),
            maxNumIterations=self.ik_max_iters,
            residualThreshold=self.ik_tol,
        )
        q_sol = np.array(sol[:len(self.movable_joints)], dtype=float)

        # 将解应用到机器人，用于检测
        for j, val in zip(self.movable_joints, q_sol):
            p.resetJointState(self.robot, j, val)

        # 检查与地面的接触（忽略 link1）
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.plane)
        for c in contacts:
            if c[3] != 1:  # linkIndexA != 1
                self.get_logger().warn(f"IK 解导致 link{c[3]} 碰撞地面，丢弃")
                return q_now  # 或者 return None
        return q_sol


    # ---------- 发送控制器 ----------
    def send_to_action(self, q_star: np.ndarray, duration_sec: float = 3.0):
        """把单点关节位姿作为目标发送给控制器（FollowJointTrajectory）"""
        traj = JointTrajectory()
        traj.joint_names = self.urdf_joint_names

        pt = JointTrajectoryPoint()
        pt.positions = q_star.tolist()
        pt.time_from_start = RosDuration(sec=int(duration_sec), nanosec=int((duration_sec % 1.0)*1e9))
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = RosDuration(sec=1)

        self.get_logger().info(f"等待 action server: {self.follow_traj_action}")
        self.client.wait_for_server()

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("控制器拒绝目标")
            return

        self.get_logger().info("目标已接受，等待执行完成...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        self.get_logger().info(f"执行完成: status={getattr(result, 'status', 'unknown')}")

    # ---------- 主流程 ----------
    def run(self):
            # 等待 joint_states
            t0 = time.time()
            while time.time() - t0 < 2.0 and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                self.mirror_joint_states_to_pybullet()
                p.stepSimulation()

            # ========= 改成圆轨迹 =========
            current_pose = p.getLinkState(self.robot, self.ee_link)[0]
            circle_traj = generate_circle_xy(center=current_pose[:2], radius=0.1, z=current_pose[2], num_points=50)

            self.follow_cartesian_traj(circle_traj, duration=10.0)

            # ========= 执行完后画图 =========
            ref = np.array(self.ref_traj)
            real = np.array(self.real_traj)

            plt.figure()
            plt.plot(ref[:,0], ref[:,1], 'r--', label="reference trajectory")
            plt.plot(real[:,0], real[:,1], 'b-', label="real trajectory")
            plt.axis('equal')
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.legend()
            plt.title("XY plane")
            plt.show()

            # 保持 GUI
            self.get_logger().info("轨迹执行完成，进入可视化保持状态（Ctrl+C退出）")
            try:
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.02)
                    p.stepSimulation()
                    time.sleep(1.0 / self.gui_rate_hz)
            except KeyboardInterrupt:
                pass




def main():
    rclpy.init()
    node = PyBulletIKVizNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        p.disconnect()


if __name__ == "__main__":
    main()
