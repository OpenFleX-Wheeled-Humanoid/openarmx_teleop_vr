#!/usr/bin/env python3
"""ROS2 adapter for OpenFlex Pico relative VR teleoperation."""

import ast
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from openarmx_arm_driver import TeleopConfig
from openarmx_arm_driver._lib.teleop_core import (
    AbsoluteTeleopInputFrame,
    PinocchioTeleopCore,
    PoseInput,
    RelativeTeleopInputFrame,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, String


class OpenArmXTeleopVRNode(Node):
    """OpenFlex ROS wrapper around the Pinocchio relative teleop core."""

    def __init__(self):
        super().__init__("openarmx_teleop_vr_node")

        self.cb_group = ReentrantCallbackGroup()
        self._declare_parameters()

        urdf_path = self.get_parameter("urdf_path").value
        if not urdf_path:
            self.get_logger().error("URDF path not provided")
            raise ValueError("urdf_path parameter is required")
        self.urdf_path = urdf_path
        self.controller_pose_mode = self._normalize_pose_mode(
            self.get_parameter("controller_pose_mode").value
        )
        self.ik_enable_override = False
        self.arm_joint_count = int(self.get_parameter("arm_joint_count").value)
        self.robot_type = self._normalize_robot_type(
            self.get_parameter("robot_type").value
        )
        # The robot end type, rather than the VR pose mode, determines the
        # controller command shape: ordinary gripper controllers expose 8
        # joints, while O6 arm controllers expose 7 joints.
        self.append_gripper_to_arm_command = self._command_includes_gripper(
            self.robot_type, self.controller_pose_mode
        )
        self.publish_separate_gripper_command = bool(
            self.get_parameter("publish_separate_gripper_command").value
        )

        self.control_rate = float(self.get_parameter("control_rate").value)
        self.grip_threshold = float(self.get_parameter("grip_threshold").value)
        self.resync_threshold_deg = float(self.get_parameter("resync_threshold_deg").value)
        self.ik_iterations = int(self.get_parameter("ik_iterations").value)
        self.sync_joint_states_each_cycle = bool(
            self.get_parameter("sync_joint_states_each_cycle").value
        )
        self.relative_stream_timeout_sec = float(
            self.get_parameter("relative_stream_timeout_sec").value
        )

        # 步长限制参数 - 逐关节配置
        max_step_joint1_2 = float(self.get_parameter("max_step_deg_joint1_2").value)
        max_step_joint3_4 = float(self.get_parameter("max_step_deg_joint3_4").value)
        max_step_joint5_7 = float(self.get_parameter("max_step_deg_joint5_7").value)

        threshold_joint1_2 = float(self.get_parameter("threshold_deg_joint1_2").value)
        threshold_joint3_4 = float(self.get_parameter("threshold_deg_joint3_4").value)
        threshold_joint5_7 = float(self.get_parameter("threshold_deg_joint5_7").value)

        # 回零模式参数
        home_step_deg = float(self.get_parameter("home_step_deg").value)
        self.max_step_rad_home = np.deg2rad(home_step_deg)

        # 构建逐关节步长限制数组（14 个关节：左臂 7 + 右臂 7）
        # 只限制 Joint 1-2，其余关节设为无穷大（不限制）
        self.max_step_rad_per_joint = np.array([
            # 左臂
            np.deg2rad(max_step_joint1_2),  # joint1 (RS04)
            np.deg2rad(max_step_joint1_2),  # joint2 (RS04)
            np.inf,  # joint3 (不限制)
            np.inf,  # joint4 (不限制)
            np.inf,  # joint5 (不限制)
            np.inf,  # joint6 (不限制)
            np.inf,  # joint7 (不限制)
            # 右臂
            np.deg2rad(max_step_joint1_2),  # joint1 (RS04)
            np.deg2rad(max_step_joint1_2),  # joint2 (RS04)
            np.inf,  # joint3 (不限制)
            np.inf,  # joint4 (不限制)
            np.inf,  # joint5 (不限制)
            np.inf,  # joint6 (不限制)
            np.inf,  # joint7 (不限制)
        ], dtype=np.float64)

        # 构建逐关节阈值数组（14 个关节：左臂 7 + 右臂 7）
        # 只对 Joint 1-2 设置阈值，其余关节设为无穷大（始终直接通过）
        self.threshold_rad_per_joint = np.array([
            # 左臂
            np.deg2rad(threshold_joint1_2),  # joint1 (RS04)
            np.deg2rad(threshold_joint1_2),  # joint2 (RS04)
            np.inf,  # joint3 (不限制)
            np.inf,  # joint4 (不限制)
            np.inf,  # joint5 (不限制)
            np.inf,  # joint6 (不限制)
            np.inf,  # joint7 (不限制)
            # 右臂
            np.deg2rad(threshold_joint1_2),  # joint1 (RS04)
            np.deg2rad(threshold_joint1_2),  # joint2 (RS04)
            np.inf,  # joint3 (不限制)
            np.inf,  # joint4 (不限制)
            np.inf,  # joint5 (不限制)
            np.inf,  # joint6 (不限制)
            np.inf,  # joint7 (不限制)
        ], dtype=np.float64)

        self.pose_mutex = threading.Lock()
        self.control_lock = threading.Lock()

        self.joint_positions = {}
        self.joint_states_received = False

        self.current_mode = None
        self._init_pose_state()
        self._set_pose_mode_matrices(self.controller_pose_mode)

        self.core_by_mode = {
            mode: self._create_core(mode) for mode in ("gripper", "hand")
        }
        self.core = self.core_by_mode[self.controller_pose_mode]
        self.get_logger().info(
            f"Pinocchio relative teleop cores initialized (mode={self.controller_pose_mode})"
        )

        self._setup_ros_io()
        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_loop,
            callback_group=self.cb_group,
        )
        self.get_logger().info(f"Node initialized - control rate: {self.control_rate} Hz")

    def _declare_parameters(self):
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("controller_pose_mode", "gripper")
        self.declare_parameter("robot_type", "gripper")
        self.declare_parameter("arm_joint_count", 7)
        self.declare_parameter("publish_separate_gripper_command", False)
        self.declare_parameter("left_gripper_cmd_topic", "")
        self.declare_parameter("right_gripper_cmd_topic", "")
        self.declare_parameter("relative_stream_timeout_sec", 0.3)
        self.declare_parameter(
            "ik_enable_override_topic", "/openarmx_teleop_vr/ik_enable_override"
        )
        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("grip_threshold", 0.5)
        self.declare_parameter("resync_threshold_deg", 5.0)
        self.declare_parameter("ik_iterations", 3)
        self.declare_parameter("sync_joint_states_each_cycle", True)

        # 步长限制参数 - 逐关节配置
        # 根据电机型号设置不同的步长限制
        self.declare_parameter("max_step_deg_joint1_2", 8.0)  # RS04 大扭矩电机
        self.declare_parameter("max_step_deg_joint3_4", 5.0)  # RS03 中扭矩电机：激进
        self.declare_parameter("max_step_deg_joint5_7", 5.0)  # RS00 小扭矩电机：激进

        # 逐关节阈值参数 - 根据电机型号设置不同的启用阈值
        self.declare_parameter("threshold_deg_joint1_2", 4.0)   # RS04 大关节：较小阈值
        self.declare_parameter("threshold_deg_joint3_4", 12.0)  # RS03 中关节：较大阈值
        self.declare_parameter("threshold_deg_joint5_7", 12.0)  # RS00 小关节：较大阈值

        # 回零模式参数
        self.declare_parameter("home_step_deg", 8.0)  # 回零模式的步长限制

        self.declare_parameter("left_pose_topic", "/pico_left_controller/pose")
        self.declare_parameter("right_pose_topic", "/pico_right_controller/pose")
        self.declare_parameter("left_grip_topic", "/pico_left_controller/grip")
        self.declare_parameter("right_grip_topic", "/pico_right_controller/grip")
        self.declare_parameter("left_trigger_topic", "/pico_left_controller/trigger")
        self.declare_parameter("right_trigger_topic", "/pico_right_controller/trigger")
        self.declare_parameter("pose_source_mode_topic", "/vr/controller_pose_mode")
        self.declare_parameter("button_a_topic", "/pico_right_controller/button_a")
        self.declare_parameter("joint_states_topic", "/joint_states")

        self.declare_parameter("left_cmd_topic", "/left_forward_position_controller/commands")
        self.declare_parameter("right_cmd_topic", "/right_forward_position_controller/commands")

        self.declare_parameter("body_anchor_offset", [0.0, -0.18, 0.08])
        self.declare_parameter("position_scale_xyz", [1.0, 1.0, 0.9])
        self.declare_parameter(
            "left_axis_matrix",
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "right_axis_matrix",
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "left_orientation_matrix",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "right_orientation_matrix",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "gripper_left_axis_matrix",
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "gripper_right_axis_matrix",
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "gripper_left_orientation_matrix",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "gripper_right_orientation_matrix",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "hand_left_axis_matrix",
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "hand_right_axis_matrix",
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "hand_left_orientation_matrix",
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        )
        self.declare_parameter(
            "hand_right_orientation_matrix",
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        )

    @staticmethod
    def _normalize_pose_mode(value):
        mode = str(value).strip().lower()
        return mode if mode in ("gripper", "hand") else "gripper"

    @staticmethod
    def _normalize_robot_type(value):
        robot_type = str(value).strip().lower()
        if robot_type not in ("gripper", "o6", "auto"):
            return "gripper"
        return robot_type

    @classmethod
    def _command_includes_gripper(cls, robot_type, pose_mode):
        normalized_type = cls._normalize_robot_type(robot_type)
        if normalized_type == "auto":
            return cls._normalize_pose_mode(pose_mode) == "gripper"
        return normalized_type == "gripper"

    def _create_core(self, mode):
        core_config = TeleopConfig(
            urdf_path=self.urdf_path,
            grip_threshold=self.grip_threshold,
            body_anchor_offset=np.asarray(
                self.get_parameter("body_anchor_offset").value, dtype=np.float64
            ),
            position_scale_xyz=np.asarray(
                self.get_parameter("position_scale_xyz").value, dtype=np.float64
            ),
            left_axis_matrix=self._matrix_from_parameter(f"{mode}_left_axis_matrix"),
            right_axis_matrix=self._matrix_from_parameter(f"{mode}_right_axis_matrix"),
            left_orientation_matrix=self._matrix_from_parameter(
                f"{mode}_left_orientation_matrix"
            ),
            right_orientation_matrix=self._matrix_from_parameter(
                f"{mode}_right_orientation_matrix"
            ),
            slow_max_step_deg=0.0,
            fast_max_step_deg=0.0,
            resync_threshold_deg=self.resync_threshold_deg,
            ik_iterations=self.ik_iterations,
            sync_joint_states_each_cycle=self.sync_joint_states_each_cycle,
            stream_timeout_sec=0.3,
        )
        return PinocchioTeleopCore(core_config)

    def _init_pose_state(self):
        self.relative_left_raw_pose = None
        self.relative_right_raw_pose = None
        self.relative_left_reference_pose = None
        self.relative_right_reference_pose = None
        self.relative_left_enabled = False
        self.relative_right_enabled = False
        self.relative_left_pose = None
        self.relative_right_pose = None
        self.relative_left_grip = 0.0
        self.relative_right_grip = 0.0
        self.relative_left_trigger = 0.0
        self.relative_right_trigger = 0.0
        self.button_a_pressed = False
        self.homing_mode = False  # 回零模式标志
        self.returning_home = False  # A键回零状态

        # Keep an inactive arm at the pose where its grip was released. The IK
        # core returns a target for both arms, but an inactive arm must not have
        # that target regenerated from its latest feedback every cycle.
        self.left_hold_command = None
        self.right_hold_command = None
        self.left_was_active = False
        self.right_was_active = False

    def _set_pose_mode_matrices(self, mode):
        self.relative_left_axis_matrix = self._matrix_from_parameter(
            f"{mode}_left_axis_matrix"
        )
        self.relative_right_axis_matrix = self._matrix_from_parameter(
            f"{mode}_right_axis_matrix"
        )
        self.relative_left_orientation_matrix = self._matrix_from_parameter(
            f"{mode}_left_orientation_matrix"
        )
        self.relative_right_orientation_matrix = self._matrix_from_parameter(
            f"{mode}_right_orientation_matrix"
        )

    def _matrix_from_parameter(self, name: str) -> np.ndarray:
        raw_value = self.get_parameter(name).value
        if isinstance(raw_value, str):
            raw_value = ast.literal_eval(raw_value)
        values = np.asarray(list(raw_value), dtype=np.float64)
        if values.size != 9:
            raise ValueError(f"{name} must contain 9 values, got {values.size}")
        return values.reshape(3, 3)

    def _pose_from_msg(self, msg: PoseStamped) -> PoseInput:
        position = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=np.float64,
        )
        orientation_xyzw = (
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        )
        timestamp = time.monotonic()
        try:
            return PoseInput(
                position=position,
                orientation_xyzw=orientation_xyzw,
                timestamp=timestamp,
            )
        except TypeError:
            pose = PoseInput()
            pose.position = position
            pose.orientation_xyzw = orientation_xyzw
            pose.timestamp = timestamp
            return pose

    def _copy_pose(self, pose: PoseInput) -> PoseInput:
        copied = PoseInput()
        copied.position = np.asarray(pose.position, dtype=np.float64).copy()
        copied.orientation_xyzw = tuple(float(value) for value in pose.orientation_xyzw)
        copied.timestamp = pose.timestamp
        return copied

    @staticmethod
    def _identity_relative_pose(timestamp=None) -> PoseInput:
        pose = PoseInput()
        pose.position = np.zeros(3, dtype=np.float64)
        pose.orientation_xyzw = (0.0, 0.0, 0.0, 1.0)
        pose.timestamp = time.monotonic() if timestamp is None else timestamp
        return pose

    @staticmethod
    def _quaternion_to_rotation_matrix(qx, qy, qz, qw):
        return np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_to_quaternion(rotation):
        matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (matrix[2, 1] - matrix[1, 2]) / scale
            qy = (matrix[0, 2] - matrix[2, 0]) / scale
            qz = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
        quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm == 0.0:
            return 0.0, 0.0, 0.0, 1.0
        quaternion /= norm
        return tuple(float(value) for value in quaternion)

    def _compute_raw_relative_pose(self, reference: PoseInput, current: PoseInput) -> PoseInput:
        reference_rotation = self._quaternion_to_rotation_matrix(*reference.orientation_xyzw)
        current_rotation = self._quaternion_to_rotation_matrix(*current.orientation_xyzw)
        delta_pose = PoseInput()
        delta_pose.position = reference_rotation.T @ (
            np.asarray(current.position, dtype=np.float64)
            - np.asarray(reference.position, dtype=np.float64)
        )
        delta_rotation = reference_rotation.T @ current_rotation
        delta_pose.orientation_xyzw = self._rotation_matrix_to_quaternion(delta_rotation)
        delta_pose.timestamp = current.timestamp
        return delta_pose

    def _map_relative_pose(self, pose, axis_matrix, orientation_matrix):
        mapped = self._copy_pose(pose)
        mapped.position = np.asarray(axis_matrix, dtype=np.float64) @ mapped.position
        rotation = self._quaternion_to_rotation_matrix(*mapped.orientation_xyzw)
        mapped_rotation = orientation_matrix @ rotation @ orientation_matrix.T
        mapped.orientation_xyzw = self._rotation_matrix_to_quaternion(mapped_rotation)
        return mapped

    def _effective_relative_enable(self, grip_value):
        return self.ik_enable_override or grip_value > self.grip_threshold

    def _update_relative_pose_state(self, arm):
        raw_pose = getattr(self, f"relative_{arm}_raw_pose")
        reference_attr = f"relative_{arm}_reference_pose"
        enabled_attr = f"relative_{arm}_enabled"
        pose_attr = f"relative_{arm}_pose"
        grip_value = getattr(self, f"relative_{arm}_grip")
        axis_matrix = getattr(self, f"relative_{arm}_axis_matrix")
        orientation_matrix = getattr(self, f"relative_{arm}_orientation_matrix")
        enabled = self._effective_relative_enable(grip_value)
        was_enabled = getattr(self, enabled_attr)
        raw_pose_fresh = (
            raw_pose is not None
            and (time.monotonic() - raw_pose.timestamp) <= self.relative_stream_timeout_sec
        )

        if not enabled or not raw_pose_fresh:
            release_pose = self._identity_relative_pose() if was_enabled else None
            setattr(self, enabled_attr, False)
            setattr(self, reference_attr, None)
            setattr(self, pose_attr, release_pose)
            return

        reference = getattr(self, reference_attr)
        if not was_enabled or reference is None:
            setattr(self, reference_attr, self._copy_pose(raw_pose))
            relative_pose = self._identity_relative_pose(raw_pose.timestamp)
        else:
            relative_pose = self._compute_raw_relative_pose(reference, raw_pose)
        setattr(self, enabled_attr, True)
        setattr(
            self,
            pose_attr,
            self._map_relative_pose(relative_pose, axis_matrix, orientation_matrix),
        )

    @staticmethod
    def _make_relative_input(
        left_pose,
        right_pose,
        left_grip,
        right_grip,
        left_trigger,
        right_trigger,
    ) -> RelativeTeleopInputFrame:
        try:
            return RelativeTeleopInputFrame(
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                left_trigger=left_trigger,
                right_trigger=right_trigger,
            )
        except TypeError:
            frame = RelativeTeleopInputFrame()
            frame.left_pose = left_pose
            frame.right_pose = right_pose
            frame.left_grip = left_grip
            frame.right_grip = right_grip
            frame.left_trigger = left_trigger
            frame.right_trigger = right_trigger
            return frame

    @staticmethod
    def _make_absolute_input(
        head_pose,
        left_pose,
        right_pose,
        left_grip,
        right_grip,
        left_trigger,
        right_trigger,
        mode_selected,
    ) -> AbsoluteTeleopInputFrame:
        try:
            return AbsoluteTeleopInputFrame(
                head_pose=head_pose,
                left_pose=left_pose,
                right_pose=right_pose,
                left_grip=left_grip,
                right_grip=right_grip,
                left_trigger=left_trigger,
                right_trigger=right_trigger,
                mode_selected=mode_selected,
            )
        except TypeError:
            frame = AbsoluteTeleopInputFrame()
            frame.head_pose = head_pose
            frame.left_pose = left_pose
            frame.right_pose = right_pose
            frame.left_grip = left_grip
            frame.right_grip = right_grip
            frame.left_trigger = left_trigger
            frame.right_trigger = right_trigger
            frame.mode_selected = mode_selected
            return frame

    def _setup_ros_io(self):
        p = self.get_parameter

        self.relative_left_pose_sub = self.create_subscription(
            PoseStamped,
            p("left_pose_topic").value,
            self._relative_left_pose_callback,
            10,
            callback_group=self.cb_group,
        )
        self.relative_right_pose_sub = self.create_subscription(
            PoseStamped,
            p("right_pose_topic").value,
            self._relative_right_pose_callback,
            10,
            callback_group=self.cb_group,
        )
        self.relative_left_grip_sub = self.create_subscription(
            Float32,
            p("left_grip_topic").value,
            self._relative_left_grip_callback,
            10,
            callback_group=self.cb_group,
        )
        self.relative_right_grip_sub = self.create_subscription(
            Float32,
            p("right_grip_topic").value,
            self._relative_right_grip_callback,
            10,
            callback_group=self.cb_group,
        )
        self.relative_left_trigger_sub = self.create_subscription(
            Float32,
            p("left_trigger_topic").value,
            self._relative_left_trigger_callback,
            10,
            callback_group=self.cb_group,
        )
        self.relative_right_trigger_sub = self.create_subscription(
            Float32,
            p("right_trigger_topic").value,
            self._relative_right_trigger_callback,
            10,
            callback_group=self.cb_group,
        )
        self.pose_source_mode_sub = self.create_subscription(
            String,
            p("pose_source_mode_topic").value,
            self._pose_source_mode_callback,
            10,
            callback_group=self.cb_group,
        )
        ik_override_qos = QoSProfile(depth=1)
        ik_override_qos.reliability = ReliabilityPolicy.RELIABLE
        ik_override_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.ik_enable_override_sub = self.create_subscription(
            Bool,
            p("ik_enable_override_topic").value,
            self._ik_enable_override_callback,
            ik_override_qos,
            callback_group=self.cb_group,
        )
        self.button_a_sub = self.create_subscription(
            Bool,
            p("button_a_topic").value,
            self._button_a_callback,
            10,
            callback_group=self.cb_group,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            p("joint_states_topic").value,
            self._joint_state_callback,
            10,
            callback_group=self.cb_group,
        )

        self.left_cmd_pub = self.create_publisher(
            Float64MultiArray, p("left_cmd_topic").value, 10
        )
        self.right_cmd_pub = self.create_publisher(
            Float64MultiArray, p("right_cmd_topic").value, 10
        )
        self.left_gripper_cmd_pub = None
        self.right_gripper_cmd_pub = None
        if self.publish_separate_gripper_command:
            left_topic = str(p("left_gripper_cmd_topic").value).strip()
            right_topic = str(p("right_gripper_cmd_topic").value).strip()
            if not left_topic or not right_topic:
                raise ValueError(
                    "Separate gripper publishing requires both gripper command topics"
                )
            self.left_gripper_cmd_pub = self.create_publisher(
                Float64MultiArray, left_topic, 10
            )
            self.right_gripper_cmd_pub = self.create_publisher(
                Float64MultiArray, right_topic, 10
            )
        self.get_logger().info("ROS2 relative teleop interfaces configured")

    def _relative_left_pose_callback(self, msg: PoseStamped):
        with self.pose_mutex:
            self.relative_left_raw_pose = self._pose_from_msg(msg)

    def _relative_right_pose_callback(self, msg: PoseStamped):
        with self.pose_mutex:
            self.relative_right_raw_pose = self._pose_from_msg(msg)

    def _relative_left_grip_callback(self, msg: Float32):
        self.relative_left_grip = max(0.0, min(1.0, float(msg.data)))

    def _relative_right_grip_callback(self, msg: Float32):
        self.relative_right_grip = max(0.0, min(1.0, float(msg.data)))

    def _relative_left_trigger_callback(self, msg: Float32):
        self.relative_left_trigger = max(0.0, min(1.0, float(msg.data)))

    def _relative_right_trigger_callback(self, msg: Float32):
        self.relative_right_trigger = max(0.0, min(1.0, float(msg.data)))

    def _pose_source_mode_callback(self, msg: String):
        mode = self._normalize_pose_mode(msg.data)
        if mode == self.controller_pose_mode:
            return
        with self.control_lock:
            self.controller_pose_mode = mode
            if self.robot_type == "auto":
                self.append_gripper_to_arm_command = self._command_includes_gripper(
                    self.robot_type, mode
                )
            self.core = self.core_by_mode[mode]
            self._set_pose_mode_matrices(mode)
            self.current_mode = None
            self.relative_left_reference_pose = None
            self.relative_right_reference_pose = None
            self.relative_left_enabled = False
            self.relative_right_enabled = False
            self.left_hold_command = None
            self.right_hold_command = None
            self.left_was_active = False
            self.right_was_active = False
            if self.joint_states_received:
                names = list(self.joint_positions.keys())
                positions = [self.joint_positions[name] for name in names]
                self.core.update_joint_states(names, positions)
        self.get_logger().info(f"VR pose source mode switched to: {mode}")

    def _ik_enable_override_callback(self, msg: Bool):
        enabled = bool(msg.data)
        if enabled != self.ik_enable_override:
            self.ik_enable_override = enabled
            self.get_logger().info(
                f"IK enable override: {'ON' if enabled else 'OFF'}"
            )

    def _button_a_callback(self, msg: Bool):
        # 边沿检测：只在按键从未按下变为按下时触发
        if msg.data and not self.button_a_pressed:
            self.returning_home = not self.returning_home
            status = "启用" if self.returning_home else "禁用"
            self.get_logger().info(f"A键按下 - 回零位模式已{status}")
        self.button_a_pressed = msg.data

    def _joint_state_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = float(pos)
        self.joint_states_received = True
        for core in self.core_by_mode.values():
            core.update_joint_states(msg.name, msg.position)

    def _current_relative_input(self) -> RelativeTeleopInputFrame:
        with self.pose_mutex:
            self._update_relative_pose_state("left")
            self._update_relative_pose_state("right")
            left_pose = self.relative_left_pose
            right_pose = self.relative_right_pose
            left_enabled = self.relative_left_enabled
            right_enabled = self.relative_right_enabled
        return self._make_relative_input(
            left_pose=left_pose,
            right_pose=right_pose,
            left_grip=1.0 if left_enabled else 0.0,
            right_grip=1.0 if right_enabled else 0.0,
            left_trigger=self.relative_left_trigger,
            right_trigger=self.relative_right_trigger,
        )

    @staticmethod
    def _disabled_absolute_input() -> AbsoluteTeleopInputFrame:
        return OpenArmXTeleopVRNode._make_absolute_input(
            head_pose=None,
            left_pose=None,
            right_pose=None,
            left_grip=0.0,
            right_grip=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            mode_selected=False,
        )

    def _control_loop(self):
        if not self.control_lock.acquire(blocking=False):
            return
        try:
            # 检查是否正在回零位
            if self.returning_home:
                current_q = self._get_current_joint_positions()
                if current_q is None:
                    return

                # 目标位置：零位（14个关节全为0）
                home_target_q = np.zeros(14, dtype=np.float64)

                # 使用回零专用步长限制
                limited_arm_q = self._limit_joint_step_home(home_target_q, current_q)

                # 检查是否到达零位（所有关节误差 < 0.01 rad）
                if np.all(np.abs(limited_arm_q - current_q) < 0.01):
                    self.returning_home = False
                    self.get_logger().info("Reached home position")

                # 发布回零命令（夹爪保持当前状态）
                left_gripper = self._get_current_gripper_command("left")
                right_gripper = self._get_current_gripper_command("right")
                self._publish_joint_commands(
                    limited_arm_q[:7].tolist(),
                    limited_arm_q[7:14].tolist(),
                    left_gripper,
                    right_gripper,
                )
                return

            # 正常遥操作模式
            relative_input = self._current_relative_input()
            result = self.core.step(relative_input, self._disabled_absolute_input())
            if result.active_mode != self.current_mode:
                self.current_mode = result.active_mode
                if self.current_mode is None:
                    self.get_logger().info("Teleop mode idle - no active relative stream")
                else:
                    self.get_logger().info(f"Teleop mode switched to: {self.current_mode}")

            if result.waiting_for_joint_states or result.calibration_required:
                return

            current_q = self._get_current_joint_positions()
            if current_q is None:
                return

            if result.target_q is None:
                if self.left_was_active:
                    self.left_hold_command = None
                if self.right_was_active:
                    self.right_hold_command = None
                self.left_was_active = False
                self.right_was_active = False
                return

            target_q = np.asarray(result.target_q, dtype=np.float64)

            # 调试：打印维度信息（首次）
            if not hasattr(self, '_dimension_logged'):
                self.get_logger().info(f"target_q shape: {target_q.shape}, current_q shape: {current_q.shape}")
                self._dimension_logged = True

            # 只对机械臂的14个关节应用步长限制（忽略夹爪等额外关节）
            arm_target_q = target_q[:14]
            limited_arm_q = self._limit_joint_step(arm_target_q, current_q)

            # 如果 target_q 包含夹爪关节，保留它们
            if len(target_q) > 14:
                limited_q = np.concatenate([limited_arm_q, target_q[14:]])
            else:
                limited_q = limited_arm_q

            left_active = bool(result.left_active)
            right_active = bool(result.right_active)

            if left_active:
                left_command = limited_q[:7].copy()
                self.left_hold_command = left_command.copy()
            else:
                if self.left_was_active:
                    self.left_hold_command = None
                left_command = None

            if right_active:
                right_command = limited_q[7:14].copy()
                self.right_hold_command = right_command.copy()
            else:
                if self.right_was_active:
                    self.right_hold_command = None
                right_command = None

            if left_active:
                left_gripper = self._map_trigger_to_gripper(self.relative_left_trigger)
                self._publish_left_command(left_command.tolist(), left_gripper)

            if right_active:
                right_gripper = self._map_trigger_to_gripper(self.relative_right_trigger)
                self._publish_right_command(right_command.tolist(), right_gripper)

            self.left_was_active = left_active
            self.right_was_active = right_active
        except Exception as exc:
            self.get_logger().error(f"Control loop error: {exc}")
        finally:
            self.control_lock.release()

    def _get_current_joint_positions(self):
        if not self.joint_states_received:
            return None
        joint_names = [
            "openarmx_left_joint1",
            "openarmx_left_joint2",
            "openarmx_left_joint3",
            "openarmx_left_joint4",
            "openarmx_left_joint5",
            "openarmx_left_joint6",
            "openarmx_left_joint7",
            "openarmx_right_joint1",
            "openarmx_right_joint2",
            "openarmx_right_joint3",
            "openarmx_right_joint4",
            "openarmx_right_joint5",
            "openarmx_right_joint6",
            "openarmx_right_joint7",
        ]
        values = []
        for name in joint_names:
            current = self.joint_positions.get(name)
            if current is None:
                return None
            values.append(float(current))
        return np.asarray(values, dtype=np.float64)

    def _limit_joint_step(self, target_q, current_q):
        """
        限制相邻控制周期之间的关节步长，提高安全性和平滑性。

        逐关节配置策略：
        - Joint 1-2 (RS04): 1.5°/cycle → 150°/s (大扭矩，保守)
        - Joint 3-4 (RS03): 2.5°/cycle → 250°/s (中扭矩，平衡)
        - Joint 5-7 (RS00): 5.0°/cycle → 500°/s (小扭矩，激进)

        智能阈值策略：
        - 小误差（< threshold）：直接通过，不限制 → 高精度定位
        - 大误差（> threshold）：限制步长 → 安全平滑运动

        参数:
            target_q: IK 求解的目标关节角 (14,)
            current_q: 当前关节角 (14,)

        返回:
            限制后的目标关节角 (14,)
        """
        if current_q is None:
            return target_q

        # 计算增量
        delta = np.asarray(target_q, dtype=np.float64) - np.asarray(current_q, dtype=np.float64)

        # 逐关节智能阈值：只对大误差应用限制
        # 小误差直接通过，避免收敛慢的问题
        apply_limit_mask = np.abs(delta) > self.threshold_rad_per_joint

        # 逐关节限制（每个关节有自己的步长限制）
        limited_delta = np.clip(delta, -self.max_step_rad_per_joint, self.max_step_rad_per_joint)

        # 选择性应用：小误差直接通过，大误差限制
        delta = np.where(apply_limit_mask, limited_delta, delta)

        return np.asarray(current_q, dtype=np.float64) + delta

    def _limit_joint_step_home(self, target_q, current_q):
        """
        回零专用步长限制：使用统一的较大步长实现快速回零。

        参数:
            target_q: 目标关节角（零位） (14,)
            current_q: 当前关节角 (14,)

        返回:
            限制后的目标关节角 (14,)
        """
        if current_q is None:
            return target_q

        # 计算增量
        delta = np.asarray(target_q, dtype=np.float64) - np.asarray(current_q, dtype=np.float64)

        # 统一步长限制（所有关节使用相同的回零步长）
        limited_delta = np.clip(delta, -self.max_step_rad_home, self.max_step_rad_home)

        return np.asarray(current_q, dtype=np.float64) + limited_delta

    @staticmethod
    def _map_trigger_to_gripper(trigger_value):
        return max(0.0, min(1.0, float(trigger_value))) * 0.04

    def _publish_joint_commands(self, left_joints, right_joints, left_gripper, right_gripper):
        self._publish_left_command(left_joints, left_gripper)
        self._publish_right_command(right_joints, right_gripper)

    def _make_arm_command(self, joints, gripper_value=None):
        values = np.asarray(list(joints), dtype=np.float64).reshape(-1)
        if values.size != self.arm_joint_count:
            raise ValueError(
                f"Expected {self.arm_joint_count} arm joints, got {values.size}"
            )
        command = values.tolist()
        if self.append_gripper_to_arm_command and gripper_value is not None:
            command.append(float(gripper_value))
        return command

    def _publish_left_command(self, left_joints, left_gripper):
        left_cmd = Float64MultiArray()
        left_cmd.data = self._make_arm_command(left_joints, left_gripper)
        self.left_cmd_pub.publish(left_cmd)
        if self.left_gripper_cmd_pub is not None:
            gripper_cmd = Float64MultiArray()
            gripper_cmd.data = [float(left_gripper)]
            self.left_gripper_cmd_pub.publish(gripper_cmd)

    def _publish_right_command(self, right_joints, right_gripper):
        right_cmd = Float64MultiArray()
        right_cmd.data = self._make_arm_command(right_joints, right_gripper)
        self.right_cmd_pub.publish(right_cmd)
        if self.right_gripper_cmd_pub is not None:
            gripper_cmd = Float64MultiArray()
            gripper_cmd.data = [float(right_gripper)]
            self.right_gripper_cmd_pub.publish(gripper_cmd)

    def _get_current_gripper_command(self, arm: str) -> float:
        joint_name = f"openarmx_{arm}_finger_joint1"
        current = self.joint_positions.get(joint_name)
        return 0.0 if current is None else float(current)


def main(args=None):
    rclpy.init(args=args)
    node = OpenArmXTeleopVRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
