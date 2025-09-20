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


# -------------------- Quaternion utils (xyzw) --------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return (q / (n if n > 0 else 1.0)).tolist()

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def quat_from_vectors(v_from, v_to):
    """最小旋转: 把 v_from 旋到 v_to 的四元数（xyzw）"""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to   = v_to   / (np.linalg.norm(v_to) + 1e-12)
    dot = float(np.dot(v_from, v_to))
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        # 反向，取任意垂直轴
        axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return [axis[0], axis[1], axis[2], 0.0]
    axis = np.cross(v_from, v_to)
    s = math.sqrt((1.0 + dot) * 2.0)
    invs = 1.0 / s
    return [axis[0]*invs, axis[1]*invs, axis[2]*invs, 0.5*s]

def mat_from_quat_xyzw(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy)],
    ], dtype=float)

def ortho_basis_from_axis(z0):
    """给定单位向量 z0，构造与其正交的单位向量 u1,u2，使 {u1,u2,z0} 构成正交基"""
    z0 = z0 / (np.linalg.norm(z0) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = np.cross(z0, ref)
    n1 = np.linalg.norm(u1)
    if n1 < 1e-9:
        ref = np.array([0.0, 0.0, 1.0])
        u1 = np.cross(z0, ref)
        n1 = np.linalg.norm(u1)
    u1 /= (n1 + 1e-12)
    u2 = np.cross(z0, u1)
    u2 /= (np.linalg.norm(u2) + 1e-12)
    return u1, u2


class PyBulletIKNode(Node):
    """
    末端 XYZ 固定，姿态在“以当前末端Z轴为轴”的圆锥面上扫描。
    - 最小旋转（无自转累积）
    - 进入/退出圆锥的平滑过渡
    - 位置与关节跳变守护
    """

    def __init__(self):
        super().__init__("pybullet_ik_cone_attitude")

        # ===== 基础配置 =====
        self.urdf_path = "examples/urdf/cr5_robot.urdf"
        self.add_search_path = pybullet_data.getDataPath()
        self.use_fixed_base = True

        self.ee_link_name = "tool0"
        self.fallback_ee_link_index = 7

        self.follow_traj_action = "/cr5_group_controller/follow_joint_trajectory"
        self.joint_states_topic = "/joint_states"

        self.name_map_ros_to_urdf: Dict[str, str] = {}

        # IK 参数
        self.ik_max_iters = 400
        self.ik_tol = 1e-6

        # ===== 轨迹参数 =====
        self.declare_parameter('cone_half_angle_deg', 10.0)
        self.declare_parameter('points_per_loop', 360)
        self.declare_parameter('duration_per_loop', 12.0)
        self.declare_parameter('num_loops', 2)
        self.declare_parameter('transition_steps', 40)      # 过渡与回正步数

        self.cone_half_angle_deg = float(self.get_parameter('cone_half_angle_deg').value)
        self.points_per_loop     = int(self.get_parameter('points_per_loop').value)
        self.duration_per_loop   = float(self.get_parameter('duration_per_loop').value)
        self.num_loops           = int(self.get_parameter('num_loops').value)
        self.transition_steps    = int(self.get_parameter('transition_steps').value)

        # ===== 守护参数 =====
        self.declare_parameter('pos_tol_m', 0.01)            # 位置容差 (m)
        self.declare_parameter('joint_jump_deg_limit', 30.0) # 关节突跳限制 (deg)

        self.pos_tol_m            = float(self.get_parameter('pos_tol_m').value)
        self.joint_jump_deg_limit = float(self.get_parameter('joint_jump_deg_limit').value)

        # ===== PyBullet 初始化 =====
        if p.isConnected():
            p.disconnect()
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(self.add_search_path)

        self.robot = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=self.use_fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
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

        # 订阅 /joint_states
        self.latest_js: Dict[str, float] = {}
        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 50)

        # Action 客户端
        self.client = ActionClient(self, FollowJointTrajectory, self.follow_traj_action)

    # ---------- 基础 ----------
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
        if self.latest_js:
            q = []
            for name, j in zip(self.urdf_joint_names, self.movable_joints):
                q.append(self.latest_js.get(name, p.getJointState(self.robot, j)[0]))
            return np.array(q, dtype=float)
        return np.array([p.getJointState(self.robot, j)[0] for j in self.movable_joints], dtype=float)

    def _sync_bullet_with_q(self, q: np.ndarray):
        for j, val in zip(self.movable_joints, q.tolist()):
            p.resetJointState(self.robot, j, val)

    # ---------- 轨迹生成（最小旋转 + 过渡/回正） ----------
    def _cone_targets_about_z0(self, z0, theta, phis, u1=None, u2=None):
        """围绕 z0 的圆锥目标Z方向序列"""
        if u1 is None or u2 is None:
            u1, u2 = ortho_basis_from_axis(z0)
        z_targets = []
        c, s = math.cos(theta), math.sin(theta)
        for phi in phis:
            zt = c * z0 + s * (u1 * math.cos(phi) + u2 * math.sin(phi))
            z_targets.append(zt / (np.linalg.norm(zt) + 1e-12))
        return z_targets, u1, u2

    def generate_transition(self, center_pos, q0_xyzw, theta, steps, u1, u2):
        """从0→theta 的倾斜过渡（phi=0 方向，即沿 u1 倾斜）"""
        q0 = quat_normalize(q0_xyzw)
        R0 = mat_from_quat_xyzw(q0)
        z0 = R0[:, 2]
        alphas = np.linspace(0.0, theta, max(2, int(steps)))
        traj = []
        last_q = None
        for a in alphas:
            zt = math.cos(a) * z0 + math.sin(a) * u1
            q_align = quat_from_vectors(z0, zt)
            q = quat_mul(q_align, q0)
            q = quat_normalize(q)
            # 四元数符号连续化
            if last_q is not None and np.dot(last_q, q) < 0:
                q = (-np.array(q)).tolist()
            traj.append((center_pos, q))
            last_q = q
        return traj

    def generate_back_transition(self, center_pos, q0_xyzw, theta, steps, u1, u2):
        """从theta→0 的回正过渡（沿 u1 方向回撤）"""
        q0 = quat_normalize(q0_xyzw)
        R0 = mat_from_quat_xyzw(q0)
        z0 = R0[:, 2]
        alphas = np.linspace(theta, 0.0, max(2, int(steps)))
        traj = []
        last_q = None
        for a in alphas:
            zt = math.cos(a) * z0 + math.sin(a) * u1
            q_align = quat_from_vectors(z0, zt)
            q = quat_mul(q_align, q0)
            q = quat_normalize(q)
            if last_q is not None and np.dot(last_q, q) < 0:
                q = (-np.array(q)).tolist()
            traj.append((center_pos, q))
            last_q = q
        return traj

    def generate_cone_attitude_one_loop(self, center_pos, q0_xyzw,
                                        half_angle_deg: float,
                                        points_per_loop: int):
        """围绕当前 z0 的最小旋转圆锥扫描（无自转），不含过渡/回正"""
        q0 = quat_normalize(q0_xyzw)
        R0 = mat_from_quat_xyzw(q0)
        z0 = R0[:, 2]
        u1, u2 = ortho_basis_from_axis(z0)

        theta = math.radians(half_angle_deg)
        phis = np.linspace(0.0, 2.0*math.pi, max(2, int(points_per_loop)), endpoint=False)

        z_targets, u1, u2 = self._cone_targets_about_z0(z0, theta, phis, u1, u2)

        traj = []
        last_q = None
        for zt in z_targets:
            q_align = quat_from_vectors(z0, zt)
            q = quat_mul(q_align, q0)
            q = quat_normalize(q)
            if last_q is not None and np.dot(last_q, q) < 0:
                q = (-np.array(q)).tolist()
            traj.append((center_pos, q))
            last_q = q

        # 闭合点
        q_end = quat_mul(quat_from_vectors(z0, z_targets[0]), q0)
        q_end = quat_normalize(q_end)
        if last_q is not None and np.dot(last_q, q_end) < 0:
            q_end = (-np.array(q_end)).tolist()
        traj.append((center_pos, q_end))

        return traj, (u1, u2)

    # ---------- IK 守护 ----------
    def _fk_link_pose(self, q_sol):
        self._sync_bullet_with_q(q_sol)
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        return np.array(ls[4]), np.array(ls[5])

    def solve_ik_guarded(self, target_pos, target_quat_xyzw, q_now):
        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=int(self.ee_link),
            targetPosition=target_pos,
            targetOrientation=target_quat_xyzw,
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=self.joint_ranges.tolist(),
            restPoses=q_now.tolist(),
            maxNumIterations=self.ik_max_iters,
            residualThreshold=self.ik_tol,
        )
        q_sol = np.array(sol[:len(self.movable_joints)], dtype=float)

        pos_fk, _ = self._fk_link_pose(q_sol)
        pos_err = np.linalg.norm(pos_fk - np.array(target_pos))
        jump = float(np.max(np.abs(np.rad2deg(q_sol - q_now))))

        if pos_err > self.pos_tol_m or jump > self.joint_jump_deg_limit:
            self.get_logger().warn(
                f"IK 不稳定: pos_err={pos_err:.4f}, jump={jump:.1f}deg → 使用上一解"
            )
            return q_now
        return q_sol

    # ---------- 执行轨迹 ----------
    def follow_loops_cone(self, traj_one_loop, num_loops: int, duration_per_loop: float):
        q_now = self.get_q_now()
        self._sync_bullet_with_q(q_now)

        pts_per_loop = len(traj_one_loop)
        traj = JointTrajectory()
        traj.joint_names = self.urdf_joint_names

        t_offset = 0.0
        for _ in range(max(1, num_loops)):
            for i, (pos, quat) in enumerate(traj_one_loop, start=1):
                q_star = self.solve_ik_guarded(pos, quat, q_now)
                q_now = q_star

                t_local = (i / pts_per_loop) * duration_per_loop
                t = t_offset + t_local

                pt = JointTrajectoryPoint()
                pt.positions = q_star.tolist()
                pt.time_from_start = RosDuration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
                traj.points.append(pt)
            t_offset += duration_per_loop

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = RosDuration(sec=1)

        self.get_logger().info(f"等待 action server: {self.follow_traj_action}")
        self.client.wait_for_server()
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if not gh or not gh.accepted:
            self.get_logger().error("控制器拒绝轨迹")
            return

        self.get_logger().info("轨迹已接受，等待执行完成...")
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        result = rf.result()
        self.get_logger().info(f"执行完成: status={getattr(result, 'status', 'unknown')}")

    # ---------- 主流程 ----------
    def run(self):
        # 等 /joint_states
        t0 = time.time()
        while time.time() - t0 < 2.0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

        # 当前姿态 → 得到中心位姿与 z0 基
        q_now = self.get_q_now()
        self._sync_bullet_with_q(q_now)
        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        center_pos = ls[4]
        q0_xyzw    = ls[5]

        # 生成圆锥主循环（无自转）
        cone_traj, (u1, u2) = self.generate_cone_attitude_one_loop(
            center_pos=center_pos,
            q0_xyzw=q0_xyzw,
            half_angle_deg=self.cone_half_angle_deg,
            points_per_loop=self.points_per_loop
        )

        # 生成过渡和回正
        theta = math.radians(self.cone_half_angle_deg)
        transition = self.generate_transition(center_pos, q0_xyzw, theta, self.transition_steps, u1, u2)
        back_transition = self.generate_back_transition(center_pos, q0_xyzw, theta, self.transition_steps, u1, u2)

        # 只在最开始加正向过渡，最后加回正
        traj_all = transition
        for loop_idx in range(self.num_loops):
            traj_all += cone_traj
        traj_all += back_transition

        # 执行
        self.follow_loops_cone(
            traj_one_loop=traj_all,
            num_loops=1,   # 已经手动拼接了所有循环，所以这里给1
            duration_per_loop=self.duration_per_loop * (self.num_loops + 2*self.transition_steps/self.points_per_loop)
        )


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
