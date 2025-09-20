# 编译该工程
colcon build --merge-install --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release"
source install/setup.bash

####### 以下需要首先启动 “https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4” 工程 #########
# 对于dodot_ws工程，首先需要启动
  # gazebo
  source /opt/ros/humble/setup.bash
  source /usr/share/gazebo/setup.bash
  source install/local_setup.sh
  export DOBOT_TYPE=cr5
  ros2 launch dobot_gazebo gazebo_moveit.launch.py

  # moveit
  source /opt/ros/humble/setup.bash
  source /usr/share/gazebo/setup.bash
  source install/local_setup.sh
  export DOBOT_TYPE=cr5
  ros2 launch dobot_moveit moveit_gazebo.launch.py
##############################################################

########## 对于本工程，目前主要完成以下功能 ###########
# 关节单点规划，执行后可在rviz和gazebo下看到相应运动，这个是回到一个比较好的初始位置关节
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
source install/setup.bash
ros2 run pymoveit2 ex_joint_goal.py --ros-args --params-file examples/config/ex_joint_goal.params.yaml
# 关节多点规划
ros2 run pymoveit2 ex_joint_goal.py --ros-args --params-file examples/config/ex_joint_multi_goal.params.yaml

# 关节流输入执行
ros2 run pymoveit2 ex_joint_goal_stream.py --ros-args \
  --params-file examples/config/ex_joint_goal_stream.params.yaml

# servo轨迹
ros2 run pymoveit2 ex_servo.py

# 根据urdf计算ik
ros2 run pymoveit2 move_to_pose.py --ros-args -p targets_yaml:=examples/config/cartesian_targets.yaml

# 执行空间轨迹运动
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.bash
source install/setup.bash
ros2 run pymoveit2 move_circle_trajectory.py
ros2 run pymoveit2 move_sine_trajectory.py

# 圆锥姿态运动，这个是应要求编写的圆锥焊接法的测试demo，可以调整圆锥度数、运行时间、插值点数、限制条件等
# 圆锥的轴线以当前末端的Z轴指向为轴，保持灵活性
ros2 run pymoveit2 axis_angle.py --ros-args \
  -p cone_half_angle_deg:=5.0 \
  -p points_per_loop:=720 \
  -p duration_per_loop:=10.0 \
  -p num_loops:=4 \
  -p transition_steps:=40 \
  -p pos_tol_m:=0.01 \
  -p joint_jump_deg_limit:=45.0





