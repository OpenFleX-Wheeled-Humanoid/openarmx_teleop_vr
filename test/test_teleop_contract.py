import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from openarmx_arm_driver._lib.teleop_core import PoseInput
from openarmx_teleop_vr.openarmx_teleop_vr_node import OpenArmXTeleopVRNode


class TeleopContractTest(unittest.TestCase):
    def test_gripper_robot_command_contains_eight_values(self):
        node = object.__new__(OpenArmXTeleopVRNode)
        node.arm_joint_count = 7
        node.append_gripper_to_arm_command = True

        command = node._make_arm_command(range(7), 0.04)

        self.assertEqual(command, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.04])

    def test_gripper_default_config_appends_gripper_to_arm_command(self):
        config_path = Path(__file__).parents[1] / "config" / "teleop_params.yaml"
        config = yaml.safe_load(config_path.read_text())
        params = config["openarmx_teleop_vr_node"]["ros__parameters"]

        self.assertEqual(params["arm_joint_count"], 7)
        self.assertEqual(params["robot_type"], "auto")
        self.assertFalse(params["publish_separate_gripper_command"])

    def test_robot_type_selects_controller_command_shape(self):
        self.assertEqual(OpenArmXTeleopVRNode._normalize_robot_type("gripper"), "gripper")
        self.assertEqual(OpenArmXTeleopVRNode._normalize_robot_type("o6"), "o6")
        self.assertEqual(OpenArmXTeleopVRNode._normalize_robot_type("auto"), "auto")
        self.assertEqual(OpenArmXTeleopVRNode._normalize_robot_type("unknown"), "gripper")
        self.assertTrue(OpenArmXTeleopVRNode._command_includes_gripper("auto", "gripper"))
        self.assertFalse(OpenArmXTeleopVRNode._command_includes_gripper("auto", "hand"))

    def test_o6_command_does_not_append_gripper(self):
        node = object.__new__(OpenArmXTeleopVRNode)
        node.arm_joint_count = 7
        node.append_gripper_to_arm_command = False

        command = node._make_arm_command(range(7), 0.04)

        self.assertEqual(command, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_relative_pose_is_mapped_after_reference_capture(self):
        node = object.__new__(OpenArmXTeleopVRNode)
        node.core = SimpleNamespace(
            _quaternion_to_rotation_matrix=OpenArmXTeleopVRNode._quaternion_to_rotation_matrix
        )

        reference = PoseInput()
        reference.position = np.array([1.0, 2.0, 3.0])
        reference.orientation_xyzw = (0.0, 0.0, 0.0, 1.0)
        reference.timestamp = 1.0

        current = PoseInput()
        current.position = np.array([1.0, 3.0, 3.0])
        current.orientation_xyzw = (
            0.0,
            0.0,
            math.sin(math.pi / 4.0),
            math.cos(math.pi / 4.0),
        )
        current.timestamp = 2.0

        relative = node._compute_raw_relative_pose(reference, current)
        mapped = node._map_relative_pose(
            relative,
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            np.eye(3),
        )

        np.testing.assert_allclose(mapped.position, [0.0, 0.0, 1.0])
        self.assertAlmostEqual(mapped.orientation_xyzw[2], math.sin(math.pi / 4.0))
        self.assertAlmostEqual(mapped.orientation_xyzw[3], math.cos(math.pi / 4.0))


if __name__ == "__main__":
    unittest.main()
