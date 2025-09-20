#!/usr/bin/env python3
from threading import Thread
import math
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from builtin_interfaces.msg import Duration as RosDuration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory

from pymoveit2 import MoveIt2
from pymoveit2.robots import dobot as robot


def _chunk_waypoints(flat_list, width):
    return [flat_list[i:i + width] for i in range(0, len(flat_list), width)]


def _interp_segment(q_from, q_to, step_rad):
    """在一段关节向量之间做线性插值（不含起点，含终点）"""
    diffs = [b - a for a, b in zip(q_from, q_to)]
    max_delta = max(abs(d) for d in diffs)
    n = max(1, math.ceil(max_delta / max(step_rad, 1e-12)))
    seg = []
    for k in range(1, n + 1):
        alpha = k / n
        seg.append([a + alpha * d for a, d in zip(q_from, diffs)])
    return seg


def _dur(sec_float: float) -> RosDuration:
    return RosDuration(sec=int(sec_float), nanosec=int((sec_float - int(sec_float)) * 1e9))


def main():
    rclpy.init()
    node = Node("ex_joint_goal_stream_action")

    # —— 轨迹/插值参数 ——
    node.declare_parameter(
        "joint_waypoints_flat",
        [0.0],
        ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY),
    )
    node.declare_parameter("waypoint_width", len(robot.joint_names()))
    node.declare_parameter("interp_step_rad", 0.003)
    node.declare_parameter("include_first_waypoint", True)  # 当不从当前开始时有意义

    # —— 控制器与时间参数 ——
    node.declare_parameter("controller_topic", "/cr5_group_controller/joint_trajectory")
    node.declare_parameter("start_delay", 0.2)   # s
    node.declare_parameter("interp_dt", 0.01)    # s

    # —— 起点来源（joint_states） ——
    node.declare_parameter("start_from_current", True)
    node.declare_parameter("joint_states_topic", "/joint_states")
    node.declare_parameter("wait_js_timeout", 3.0)  # s

    # —— Action 相关 ——
    node.declare_parameter("action_server_timeout", 5.0)  # s 等待 action server
    node.declare_parameter("result_timeout", 0.0)         # s 结果超时（<=0 表示不限）
    node.declare_parameter("goal_time_tolerance", 0.2)    # s 允许到达容差

    # MoveIt 仅用于 joint_names、基本配置
    cbg = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=cbg,
    )

    # 执行器线程
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()
    node.create_rate(1.0).sleep()

    # 读取参数
    waypoints_flat = node.get_parameter("joint_waypoints_flat").get_parameter_value().double_array_value
    width          = node.get_parameter("waypoint_width").get_parameter_value().integer_value
    step_rad       = node.get_parameter("interp_step_rad").get_parameter_value().double_value
    include_first  = node.get_parameter("include_first_waypoint").get_parameter_value().bool_value
    controller_topic = node.get_parameter("controller_topic").get_parameter_value().string_value
    start_delay    = node.get_parameter("start_delay").get_parameter_value().double_value
    dt             = node.get_parameter("interp_dt").get_parameter_value().double_value

    start_from_current = node.get_parameter("start_from_current").get_parameter_value().bool_value
    js_topic       = node.get_parameter("joint_states_topic").get_parameter_value().string_value
    js_timeout     = node.get_parameter("wait_js_timeout").get_parameter_value().double_value

    action_server_timeout = node.get_parameter("action_server_timeout").get_parameter_value().double_value
    result_timeout        = node.get_parameter("result_timeout").get_parameter_value().double_value
    goal_tol_sec          = node.get_parameter("goal_time_tolerance").get_parameter_value().double_value

    # 校验输入
    if width <= 0 or len(waypoints_flat) % width != 0:
        node.get_logger().error("joint_waypoints_flat 长度必须能被 waypoint_width 整除且 width>0")
        rclpy.shutdown(); return

    if width != len(robot.joint_names()):
        node.get_logger().warn(
            f"waypoint_width({width}) 与机器人关节数({len(robot.joint_names())})不一致；"
            "确保控制器理解该子集与顺序。"
        )

    wps = _chunk_waypoints(list(waypoints_flat), width)
    if len(wps) < 1:
        node.get_logger().warn("未给出路点")
        rclpy.shutdown(); return

    # —— 读取当前 joint_states 作为起点（可选） ——
    start_q = None
    if start_from_current:
        latest = {}
        required = list(robot.joint_names())

        def js_cb(msg: JointState):
            # 记录最新位置
            for n, p in zip(msg.name, msg.position):
                latest[n] = p

        sub = node.create_subscription(JointState, js_topic, js_cb, 10)
        t0 = time.time()
        while time.time() - t0 < js_timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            if all(n in latest for n in required):
                start_q = [latest[n] for n in required]
                break

        if start_q is None:
            node.get_logger().warn(
                f"{js_timeout:.1f}s 内未从 {js_topic} 收齐关节状态，将直接从首路点开始（可能瞬移）。"
            )
        else:
            node.get_logger().info(f"从 joint_states 获取当前起点：{['%.4f'%v for v in start_q]}")

    # —— 生成稠密轨迹 ——（当前→首路点）+（各路点之间）
    dense = []
    if start_q is not None:
        dense.extend(_interp_segment(start_q, wps[0], step_rad))
        for i in range(len(wps) - 1):
            dense.extend(_interp_segment(wps[i], wps[i + 1], step_rad))
    else:
        if include_first:
            dense.append(wps[0])
        for i in range(len(wps) - 1):
            dense.extend(_interp_segment(wps[i], wps[i + 1], step_rad))

    if not dense:
        node.get_logger().warn("插值后为空（可能所有路点彼此相同），不发送。")
        rclpy.shutdown(); return

    node.get_logger().info(
        f"原始路点: {len(wps)}，插值后轨迹点数: {len(dense)}，步长: {step_rad:.6f}rad，dt: {dt:.3f}s"
    )

    # —— 构造 JointTrajectory ——（time_from_start 从 start_delay 起步）
    traj = JointTrajectory()
    traj.joint_names = robot.joint_names()
    tfs = max(start_delay, 0.0)
    for q in dense:
        pt = JointTrajectoryPoint()
        pt.positions = q
        pt.time_from_start = _dur(tfs)
        traj.points.append(pt)
        tfs += dt
    # 使用 0 stamp：以接收时刻为起点，避免系统时钟/仿真时钟不一致
    traj.header.stamp = Time(seconds=0).to_msg()

    # —— Action 客户端 —— 发送并等待结果
    # 从话题名推导 action 名：/xxx/joint_trajectory -> /xxx/follow_joint_trajectory
    if controller_topic.endswith("/joint_trajectory"):
        action_name = controller_topic[: -len("/joint_trajectory")] + "/follow_joint_trajectory"
    else:
        action_name = controller_topic.rstrip("/") + "/follow_joint_trajectory"

    client = ActionClient(node, FollowJointTrajectory, action_name)
    node.get_logger().info(f"等待 Action 服务器：{action_name} （{action_server_timeout:.1f}s 超时）...")
    if not client.wait_for_server(timeout_sec=action_server_timeout):
        node.get_logger().error("Action 服务器不可用。请确认控制器已启动且 action 名称正确。")
        rclpy.shutdown(); return

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = traj
    goal.goal_time_tolerance = _dur(goal_tol_sec)

    node.get_logger().info(
        f"发送目标：点数={len(traj.points)}，首点延迟={start_delay:.3f}s，预计总时长≈{(start_delay + dt*len(dense)):.2f}s"
    )
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().error("FollowJointTrajectory 目标被拒绝。")
        rclpy.shutdown(); return

    node.get_logger().info("目标已接受，等待执行完成...")

    result_future = goal_handle.get_result_async()
    # 带超时地等待结果（0 表示不限时）
    t_start = time.time()
    while not result_future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if result_timeout > 0.0 and (time.time() - t_start) > result_timeout:
            node.get_logger().warn(f"等待结果超时（{result_timeout:.1f}s），请求取消目标...")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel_future)
            node.get_logger().warn("已发送取消请求；脚本退出。")
            rclpy.shutdown(); return

    result = result_future.result()
    # result.status 为 GoalStatus 状态码；result.result.error_code / error_string 由控制器提供
    try:
        err = getattr(result.result, "error_code", 0)
        err_str = getattr(result.result, "error_string", "")
        node.get_logger().info(f"执行完成：status={result.status}, error_code={err}, info='{err_str}'")
    except Exception:
        node.get_logger().info(f"执行完成：status={result.status}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
