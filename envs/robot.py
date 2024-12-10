import pybullet as p
import math
from collections import namedtuple
from helper.utilities import logger
import numpy as np
from gymnasium import spaces


class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        """
        Loads the robot model by initializing robot components and parsing joint info.

        This method is intended to be called after the robot is created to initialize
        all necessary components and configurations.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        logger.info(self.joints)

    def step_simulation(self):
        """
        Step the simulation forward.

        This method should be overridden by a specific robot class to define the
        behavior for stepping the simulation.

        Raises:
            RuntimeError: If the method is not overridden by the subclass.
        """
        raise RuntimeError(
            "`step_simulation` method of RobotBase Class should be hooked by the environment."
        )

    def __parse_joint_info__(self):
        """
        Parse joint information and store it in the robot.

        This method retrieves joint information from the robot, including joint types,
        limits, forces, and velocities. It also determines which joints are controllable.

        It updates the following attributes:
            - joints
            - controllable_joints
            - arm_controllable_joints
            - arm_lower_limits
            - arm_upper_limits
            - arm_joint_ranges
        """
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "damping",
                "friction",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[
                2
            ]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(
                    self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
            info = jointInfo(
                jointID,
                jointName,
                jointType,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                controllable,
            )
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]

        self.arm_lower_limits = [
            info.lowerLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]
        self.arm_upper_limits = [
            info.upperLimit for info in self.joints if info.controllable
        ][: self.arm_num_dofs]
        self.arm_joint_ranges = [
            info.upperLimit - info.lowerLimit
            for info in self.joints
            if info.controllable
        ][: self.arm_num_dofs]

    def __init_robot__(self):
        """
        Initialize the robot. This method must be implemented by a subclass.

        Raises:
            NotImplementedError: If this method is not implemented in the subclass.
        """
        raise NotImplementedError

    def __post_load__(self):
        """
        Perform post-load actions after loading the robot model.

        This method can be overridden by subclasses to perform additional setup or configuration.
        """
        pass

    def reset(self):
        """
        Reset the robot's arm and gripper to their initial positions.

        This will call the `reset_arm` and `reset_gripper` methods to return the robot
        to its default configuration.
        """
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        Reset the arm to its rest positions.

        This method sets each joint of the arm to its predefined rest position.
        """
        for rest_pose, joint_id in zip(
            self.arm_rest_poses, self.arm_controllable_joints
        ):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        """
        Reset the gripper to its open position.

        This calls the `open_gripper` method to set the gripper to its maximum open length.
        """
        self.open_gripper()

    def get_joint_angles(self):
        joint_angles = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)

            joint_angles.append(joint_state[0])

        return joint_angles

    def open_gripper(self):
        """
        Open the gripper to its maximum opening.

        This method calls `move_gripper` with the maximum value of the gripper range.
        """
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        """
        Close the gripper to its minimum opening.

        This method calls `move_gripper` with the minimum value of the gripper range.
        """
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action, control_method):
        """
        Move the End-Effector (EE) to a new position.

        Args:
            action (list or tuple): The action specifying the target position.
                - If `control_method` is "end", `action` should be [x, y, z, roll, pitch, yaw].
                - If `control_method` is "joint", `action` should be a list of joint positions.
            control_method (str): The control method used to move the EE. It can be either:
                - "end": Move the EE using Inverse Kinematics (IK).
                - "joint": Move the EE by directly controlling the joints.
        """
        assert control_method in ("joint", "end")
        if control_method == "end":
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(
                self.id,
                self.eef_id,
                pos,
                orn,
                self.arm_lower_limits,
                self.arm_upper_limits,
                self.arm_joint_ranges,
                self.arm_rest_poses,
                maxNumIterations=20,
            )
        elif control_method == "joint":
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.VELOCITY_CONTROL,
                targetVelocity=joint_poses[i],
                force=self.joints[joint_id].maxForce,
                maxVelocity=self.joints[joint_id].maxVelocity,
            )

    def move_gripper(self, open_length):
        """
        Move the gripper to the specified opening length.

        Args:
            open_length (float): The target opening length for the gripper.
        """
        raise NotImplementedError

    def get_joint_obs(self):
        """
        Get the joint observations of the robot, including the positions and velocities
        of all controllable joints and the position of the end-effector.

        Returns:
            dict: A dictionary containing:
                - positions (list): A list of joint positions for all controllable joints.
                - velocities (list): A list of joint velocities for all controllable joints.
                - ee_pos (tuple): A tuple representing the position of the end-effector (EE)
                  in the world frame (x, y, z).
        """
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)


class Panda(RobotBase):
    def __init_robot__(self):
        # define the robot
        # see https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.eef_id = 11
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.id = p.loadURDF(
            "./assets/urdf/panda.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.gripper_range = [0, 0.04]
        # create a constraint to keep the fingers centered
        c = p.createConstraint(
            self.id,
            9,
            self.id,
            10,
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_gripper(self, open_length):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in [9, 10]:
            p.setJointMotorControl2(
                self.id, i, p.POSITION_CONTROL, open_length, force=20
            )


class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [
            -1.5690622952052096,
            -1.5446774605904932,
            1.343946009733127,
            -1.3708613585093699,
            -1.5707970583733368,
            0.0009377758247187636,
        ]
        self.id = p.loadURDF(
            "./assets/urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.gripper_range = [0, 0.085]

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        normalized_length = (open_length - 0.010) / 0.1143
        normalized_length = np.clip(normalized_length, -1, 1)
        open_angle = 0.715 - math.asin(normalized_length)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=self.joints[self.mimic_parent_id].maxVelocity,
        )

    def get_gripper_state(self):
        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        return gripper_state[0]

    # def get_joint_obs(self):
    #     joint_obs = []
    #     for joint in self.joints:
    #         joint_state = p.getJointState(self.id, joint.id)
    #         joint_position = joint_state[0]
    #         joint_velocity = joint_state[1]
    #         joint_obs.append((joint_position, joint_velocity))
    #     return joint_obs

    def get_state(self):
        """
        Returns the current state of the robot. This could include joint positions,
        joint velocities, and the position of the end-effector.
        """
        joint_state = self.get_joint_obs()
        state = (
            joint_state["positions"]
            + joint_state["velocities"]
            + list(joint_state["ee_pos"])
        )
        return np.array(state)

    def get_action_space(self):
        """
        Returns the dimension of the action space.
        In this case, let's assume the action space consists of the joint torques
        for each controllable joint in the robot.
        """
        print(self.controllable_joints)
        return len(self.controllable_joints) + 1


class UR5Robotiq140(UR5Robotiq85):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [
            -1.5690622952052096,
            -1.5446774605904932,
            1.343946009733127,
            -1.3708613585093699,
            -1.5707970583733368,
            0.0009377758247187636,
        ]
        self.id = p.loadURDF(
            "./assets/urdf/ur5_robotiq_140.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.gripper_range = [0, 0.085]
        # TODO: It's weird to use the same range and the same formula to calculate open_angle as Robotiq85.

    def __post_load__(self):
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1,
        }
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
