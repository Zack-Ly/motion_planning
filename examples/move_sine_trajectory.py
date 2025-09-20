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


def norm_quat_xyzw(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return (q / (n if n > 0 else 1.0)).tolist()


class PyBulletIKNode(Node):
    """
    简化版（无可视化）：
    - PyBullet(DIRECT) 做 IK/FK
    - 从当前末端位姿出发，Y 轴为中心做往返扫描；Z 按 Y 进度正弦起伏
    - 单圈闭合（回到中心），按 num_loops 循环
    - 整段打包成 FollowJointTrajectory 发送
    """

    def __init__(self):
        super().__init__("pybullet_ik_sine_traj")

        # ===== 基础配置 =====
        self.urdf_path = "examples/urdf/cr5_robot.urdf"
        self.add_search_path = pybullet_data.getDataPath()
        self.use_fixed_base = True

        self.ee_link_name = "tool0"      # 末端 link 名称
        self.fallback_ee_link_index = 7  # 找不到名字时的回退索引

        self.follow_traj_action = "/cr5_group_controller/follow_joint_trajectory"
        self.joint_states_topic = "/joint_states"

        # 如 ROS 名与 URDF 名不一致，可在此映射（读 joint_states 用）
        self.name_map_ros_to_urdf: Dict[str, str] = {
            # "ros_joint_1": "urdf_joint_1",
        }

        # IK 参数
        self.ik_max_iters = 300
        self.ik_tol = 1e-6

        # ===== 轨迹参数（可 ROS2 参数覆盖） =====
        # 扫描总长（以中心为 0，从 -length/2 到 +length/2）
        self.declare_parameter('length', 0.20)          # m
        # Z 方向正弦振幅
        self.declare_parameter('amplitude', 0.05)       # m
        # 在整段 length 内的正弦周期数（整数更自然）
        self.declare_parameter('cycles', 2)
        # 每条“腿”（中心→端点）的采样点数（总点数≈4N）
        self.declare_parameter('points_per_leg', 80)
        # 单圈总时长（s）
        self.declare_parameter('duration_per_loop', 8.0)
        # 循环圈数
        self.declare_parameter('num_loops', 3)

        self.length           = float(self.get_parameter('length').value)
        self.amplitude        = float(self.get_parameter('amplitude').value)
        self.cycles           = int(self.get_parameter('cycles').value)
        self.points_per_leg   = int(self.get_parameter('points_per_leg').value)
        self.duration_per_loop= float(self.get_parameter('duration_per_loop').value)
        self.num_loops        = int(self.get_parameter('num_loops').value)

        # ===== PyBullet(DIRECT) 初始化 =====
        if p.isConnected():
            p.disconnect()
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(self.add_search_path)

        self.robot = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # 关节与限位
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
                    lo, hi = -math.pi, math.pi
                self.lower_limits.append(lo)
                self.upper_limits.append(hi)
                self.joint_ranges.append(hi - lo)

        self.lower_limits = np.array(self.lower_limits)
        self.upper_limits = np.array(self.upper_limits)
        self.joint_ranges = np.array(self.joint_ranges)

        # 末端 link 索引
        idx = self._find_link_index_by_name(self.ee_link_name)
        self.ee_link = int(idx) if idx is not None else int(self.fallback_ee_link_index)

        self.get_logger().info(
            f"URDF joints={len(self.movable_joints)}; tip_link index={self.ee_link}; "
            f"joint_names={self.urdf_joint_names}"
        )

        # 订阅 /joint_states
        self.latest_js: Dict[str, float] = {}
        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 50)

        # Action 客户端
        self.client = ActionClient(self, FollowJointTrajectory, self.follow_traj_action)

    # ---------- 工具 ----------
    def _find_link_index_by_name(self, link_name: str) -> Optional[int]:
        if not link_name:
            return None
        for j in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, j)[12].decode("utf-8") == link_name:
                return j
        return None

    def _joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            urdf_name = self.name_map_ros_to_urdf.get(name, name)
            self.latest_js[urdf_name] = float(pos)

    def get_q_now(self) -> np.ndarray:
        """优先用 /joint_states；否则读 Bullet 当前值"""
        if self.latest_js:
            q = []
            for name, j in zip(self.urdf_joint_names, self.movable_joints):
                q.append(self.latest_js.get(name, p.getJointState(self.robot, j)[0]))
            return np.array(q, dtype=float)
        return np.array([p.getJointState(self.robot, j)[0] for j in self.movable_joints], dtype=float)

    def _sync_bullet_with_q(self, q: np.ndarray):
        for j, val in zip(self.movable_joints, q.tolist()):
            p.resetJointState(self.robot, j, val)

    # ---------- 单圈轨迹：中心→+Y端→−Y端→中心 ----------
    def generate_one_loop(self, start_pos, start_quat_xyzw,
                          length: float, amplitude: float,
                          cycles: int, points_per_leg: int):
        """
        start_pos: [x0, y0, z0]（“中心点”= 起点）
        Z = z0 + A * sin(2π * cycles * (s + L/2) / L), 其中 s ∈ [-L/2, +L/2]
        扫描顺序：0→+L/2 → +L/2→-L/2 → -L/2→0  （闭合回到中心）
        """
        x0, y0, z0 = map(float, start_pos)
        L = float(length)
        half = 0.5 * L
        quat = norm_quat_xyzw(start_quat_xyzw)

        # 段 1：中心 -> +端（不含端点，避免重复）
        s1 = np.linspace(0.0, +half, points_per_leg, endpoint=False)
        # 段 2：+端 -> -端（穿过中心；不含 -端，避免重复）
        s2 = np.linspace(+half, -half, 2 * points_per_leg, endpoint=False)
        # 段 3：-端 -> 中心（含中心）
        s3 = np.linspace(-half, 0.0, points_per_leg + 1, endpoint=True)

        s_all = np.concatenate([s1, s2, s3])

        traj = []
        for s in s_all:
            y = y0 + s
            # 让中心处（s=0）Z 回到 z0（sin(kπ)=0），cycles 为整数时尤自然
            phase = 2.0 * math.pi * (cycles * (s + half) / (L if L > 1e-9 else 1.0))
            z = z0 + amplitude * math.sin(phase)
            traj.append(([x0, y, z], quat))
        return traj  # [(pos, quat), ...]

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
        return q_sol

    # ---------- 组装并发送多圈轨迹 ----------
    def follow_loops(self, traj_one_loop, num_loops: int, duration_per_loop: float):
        """
        traj_one_loop: 单圈的笛卡尔点 [(pos, quat), ...]（闭合回到中心）
        """
        q_now = self.get_q_now()
        self._sync_bullet_with_q(q_now)

        pts_per_loop = len(traj_one_loop)
        total_pts = pts_per_loop * max(1, num_loops)

        traj = JointTrajectory()
        traj.joint_names = self.urdf_joint_names

        t_offset = 0.0
        for loop_idx in range(max(1, num_loops)):
            for i, (pos, quat) in enumerate(traj_one_loop, start=1):
                q_star = self.solve_ik(pos, quat, q_now)
                q_now = q_star
                self._sync_bullet_with_q(q_now)

                # 均匀定时：本圈内的相对时刻
                t_local = (i / pts_per_loop) * duration_per_loop
                t = t_offset + t_local

                pt = JointTrajectoryPoint()
                pt.positions = q_star.tolist()
                pt.time_from_start = RosDuration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
                traj.points.append(pt)

            t_offset += duration_per_loop  # 下圈时间偏移

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = RosDuration(sec=1)

        self.get_logger().info(f"等待 action server: {self.follow_traj_action}")
        self.client.wait_for_server()

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("控制器拒绝轨迹")
            return

        self.get_logger().info("轨迹已接受，等待执行完成...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        self.get_logger().info(f"执行完成: status={getattr(result, 'status', 'unknown')}")

    # ---------- 主流程 ----------
    def run(self):
        # 等待 /joint_states
        t0 = time.time()
        while time.time() - t0 < 2.0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

        # 同步到当前关节、读取“起点=中心”的末端 link 原点位姿
        q_now = self.get_q_now()
        self._sync_bullet_with_q(q_now)
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        center_pos  = ls[4]  # worldLinkFramePosition（link 原点）
        center_quat = ls[5]  # worldLinkFrameOrientation (xyzw)

        # 单圈轨迹（起点=中心，闭合回中心）
        traj_one_loop = self.generate_one_loop(
            start_pos=center_pos,
            start_quat_xyzw=center_quat,
            length=self.length,
            amplitude=self.amplitude,
            cycles=self.cycles,
            points_per_leg=self.points_per_leg
        )

        # 循环执行 num_loops 圈
        self.follow_loops(traj_one_loop, num_loops=self.num_loops, duration_per_loop=self.duration_per_loop)


def main():
    rclpy.init()
    node = PyBulletIKNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
