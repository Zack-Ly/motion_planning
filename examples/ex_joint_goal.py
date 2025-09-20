#!/usr/bin/env python3
from threading import Thread
import math

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType


from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import dobot as robot


def _chunk_waypoints(flat_list, width):
    return [flat_list[i:i + width] for i in range(0, len(flat_list), width)]


def main():
    rclpy.init()

    node = Node("ex_joint_goal")

    # --- 原有参数（保持不变） ---
    node.declare_parameter(
        "joint_positions",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    node.declare_parameter("synchronous", True)
    node.declare_parameter("cancel_after_secs", 0.0)
    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")

    # --- 新增参数（用于多点顺序执行） ---
    # 展平后的关节路点：显式声明为 DOUBLE_ARRAY（即 float 列表）
    node.declare_parameter(
        "joint_waypoints_flat",
        [0.0],  # 非空，确保被当成 DOUBLE_ARRAY
        ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY),
    )
    # 每个路点的关节数（默认等于实际关节数）
    node.declare_parameter("waypoint_width", len(robot.joint_names()))
    # 可选：每个路点之间的停顿时间（秒）
    node.declare_parameter("pause_between_waypoints", 0.0)

    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )
    moveit2.planner_id = node.get_parameter("planner_id").get_parameter_value().string_value

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    moveit2.max_velocity = 0.5
    moveit2.max_acceleration = 0.5

    # 读取参数
    joint_positions = node.get_parameter("joint_positions").get_parameter_value().double_array_value
    synchronous = node.get_parameter("synchronous").get_parameter_value().bool_value
    cancel_after_secs = node.get_parameter("cancel_after_secs").get_parameter_value().double_value
    waypoints_flat = node.get_parameter("joint_waypoints_flat").get_parameter_value().double_array_value
    waypoint_width = node.get_parameter("waypoint_width").get_parameter_value().integer_value
    pause_between = node.get_parameter("pause_between_waypoints").get_parameter_value().double_value

    # --- 如果提供了多路点，则按顺序执行 ---
    if len(waypoints_flat) > 0:
        if waypoint_width <= 0:
            node.get_logger().error("waypoint_width 必须为正整数")
            rclpy.shutdown()
            executor_thread.join()
            exit(1)

        if len(waypoints_flat) % waypoint_width != 0:
            node.get_logger().error(
                f"joint_waypoints_flat 的长度({len(waypoints_flat)})不能被 waypoint_width({waypoint_width})整除"
            )
            rclpy.shutdown()
            executor_thread.join()
            exit(1)

        if waypoint_width != len(robot.joint_names()):
            node.get_logger().warn(
                f"waypoint_width({waypoint_width}) 与机器人关节数({len(robot.joint_names())})不一致，"
                "请确认你是有意为之（例如仅控制子集）。"
            )

        waypoints = _chunk_waypoints(list(waypoints_flat), waypoint_width)
        node.get_logger().info(f"将顺序执行 {len(waypoints)} 个关节目标")

        for idx, wp in enumerate(waypoints, start=1):
            node.get_logger().info(f"[{idx}/{len(waypoints)}] Moving to waypoint: {wp}")
            moveit2.move_to_configuration(wp)
            # 为保证“一个点完成后再到下一个”，这里总是等待执行完成
            moveit2.wait_until_executed()

            if pause_between > 0.0 and idx < len(waypoints):
                node.get_logger().info(f"暂停 {pause_between:.3f}s")
                node.create_rate(1.0 / max(pause_between, 1e-9)).sleep()

        rclpy.shutdown()
        executor_thread.join()
        exit(0)

    # --- 否则保留原有单点模式 ---
    node.get_logger().info(f"Moving to {{joint_positions: {list(joint_positions)}}}")
    moveit2.move_to_configuration(joint_positions)
    if synchronous:
        moveit2.wait_until_executed()
    else:
        print("Current State: " + str(moveit2.query_state()))
        rate = node.create_rate(10)
        while moveit2.query_state() != MoveIt2State.EXECUTING:
            rate.sleep()

        print("Current State: " + str(moveit2.query_state()))
        future = moveit2.get_execution_future()

        if cancel_after_secs > 0.0:
            sleep_time = node.create_rate(cancel_after_secs)
            sleep_time.sleep()
            print("Cancelling goal")
            moveit2.cancel_execution()

        while not future.done():
            rate.sleep()

        print("Result status: " + str(future.result().status))
        print("Result error code: " + str(future.result().result.error_code))

    rclpy.shutdown()
    executor_thread.join()
    exit(0)


if __name__ == "__main__":
    main()
