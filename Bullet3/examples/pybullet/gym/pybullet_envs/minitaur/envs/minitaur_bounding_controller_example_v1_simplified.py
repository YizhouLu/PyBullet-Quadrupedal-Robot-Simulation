#This script test the bounding gait generated in MATLAB

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
os.sys.path.insert(0, parentdir)

import tensorflow as tf
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_float("motor_kp", 10, "The position gain of the motor.")
flags.DEFINE_float("motor_kd", 0.15, "The speed gain of the motor.")
flags.DEFINE_float("control_latency", 0.02, "The latency between sensor measurement and action execution the robot.")
flags.DEFINE_string("log_path", None, "The directory to write the log file.")

import gym, pybullet
import numpy as np, collections, re, math, time
from gym.utils import seeding

import pybullet_data
from pybullet_envs.minitaur.envs import bullet_client as BC
from pybullet_envs.minitaur.envs import minitaur_logging
from pybullet_envs.minitaur.envs import minitaur_logging_pb2

import scipy.io as sio
mat1 = sio.loadmat('vel_07_InitialCondition_to_VS.mat')
sorted(mat1.keys())
IP_Torso_Position = mat1['init_pos']
IP_Torso_Rotation = mat1['init_rot']
IP_Torso_Velocity = mat1['init_vel']
IP_Torso_AngularV = mat1['init_avel']
IP_Joint_Position = mat1['init_joint']
IP_Joint_Velocity = mat1['init_jvel']
INIT_POSITION = [IP_Torso_Position[0][0],IP_Torso_Position[0][1],IP_Torso_Position[0][2]]
INIT_RACK_POSITION = [0, 0, 1]
INIT_ORIENTATION = [IP_Torso_Rotation[0][0],IP_Torso_Rotation[0][1],IP_Torso_Rotation[0][2],IP_Torso_Rotation[0][3]]
INIT_VELOCITY = [IP_Torso_Velocity[0],IP_Torso_Velocity[1],IP_Torso_Velocity[2]]
INIT_ANGULAR_VELOCITY = [IP_Torso_AngularV[0],IP_Torso_AngularV[1],IP_Torso_AngularV[2]]

mat2 = sio.loadmat('vel_07_Inputs_to_VS.mat')
sorted(mat2.keys())
Inputs_Joint_Position = mat2['int_joint_position']
Inputs_Joint_Velocity = mat2['int_joint_velocity']

mat3 = sio.loadmat('vel_07_States_to_VS.mat')
sorted(mat3.keys())
base_velocity_x_desire = mat3['base_velocity_x']
base_angle_pitch_desire = mat3['base_angle_pitch']

import matplotlib.pyplot as plt

# MOTOR CONSTANT
MOTOR_VOLTAGE = 16.0
MOTOR_RESISTANCE = 0.186
MOTOR_TORQUE_CONSTANT = 0.0954
MOTOR_VISCOUS_DAMPING = 0
NUM_MOTORS = 8
VOLTAGE_CLIPPING = 50
OBSERVED_TORQUE_LIMIT = 5.7
MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING + MOTOR_TORQUE_CONSTANT)

# MINITAUR CONSTANT
_CHASSIS_NAME_PATTERN = re.compile(r"chassis\D*center")
_MOTOR_NAME_PATTERN = re.compile(r"motor\D*joint")
_KNEE_NAME_PATTERN = re.compile(r"knee\D*")
MOTOR_NAMES = [ "motor_front_leftL_joint",  "motor_front_leftR_joint",  "motor_back_leftL_joint",  "motor_back_leftR_joint",
                "motor_front_rightL_joint", "motor_front_rightR_joint", "motor_back_rightL_joint", "motor_back_rightR_joint"]
LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]

# ENVIRONMENT CONSTANT
SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_URDF_VERSION = "default"
NUM_SIMULATION_ITERATION_STEPS = 300

# CONTROLLER CONSTANT
NUM_MOTORS = 8
NUM_LEGS = 4
UPPER_LEG_LEN = 0.112
LOWER_SHORT_LEG_LEN = 0.199
LOWER_LONG_LEG_LEN = 0.2315

# MAIN CONSTANT
motor_front_leftL_position_desire = []
motor_front_leftL_position_actual = []
motor_front_leftR_position_desire = []
motor_front_leftR_position_actual = []
motor_front_leftL_velocity_desire = []
motor_front_leftL_velocity_actual = []
motor_front_leftR_velocity_desire = []
motor_front_leftR_velocity_actual = []
motor_back_leftL_position_desire = []
motor_back_leftL_position_actual = []
motor_back_leftR_position_desire = []
motor_back_leftR_position_actual = []
motor_back_leftL_velocity_desire = []
motor_back_leftL_velocity_actual = []
motor_back_leftR_velocity_desire = []
motor_back_leftR_velocity_actual = []
base_velocity_x_actual = []
base_angle_pitch_actual = []

class MotorModel(object):
    def __init__(self, torque_control_enabled = False, kp = 1.2, kd = 0):
        self._torque_control_enabled = torque_control_enabled
        self._kp = kp
        self._kd = kd
        self._resistance = MOTOR_RESISTANCE
        self._voltage = MOTOR_VOLTAGE
        self._torque_constant = MOTOR_TORQUE_CONSTANT
        self._viscous_damping = MOTOR_VISCOUS_DAMPING
        self._current_table = [0, 10, 20, 30, 40, 50, 60]
        self._torque_table = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]
        self._strength_ratios = [1.0] * NUM_MOTORS

    def convert_to_torque(self, motor_commands, true_motor_angle, velocity_commands, true_motor_velocity, kp = None, kd = None):
        if self._torque_control_enabled:
            pwm = motor_commands
        else:
            if kp is None:
                kp = np.full(NUM_MOTORS, self._kp)
            if kd is None:
                kd = np.full(NUM_MOTORS, self._kd)
            pwm = kp * (motor_commands - true_motor_angle) + kd * (velocity_commands - true_motor_velocity)
        pwm = np.clip(pwm, -1.0, 1.0)
        return self._convert_to_torque_from_pwm(pwm, true_motor_velocity)

    def _convert_to_torque_from_pwm(self, pwm, true_motor_velocity):
        observed_torque = np.clip(
            self._torque_constant * (np.asarray(pwm) * self._voltage / self._resistance),
            -OBSERVED_TORQUE_LIMIT,
            OBSERVED_TORQUE_LIMIT)
        voltage_net = np.clip(
            np.asarray(pwm) * self._voltage - (self._torque_constant + self._viscous_damping) * np.asarray(true_motor_velocity),
            -VOLTAGE_CLIPPING,
            VOLTAGE_CLIPPING)
        current = voltage_net / self._resistance
        current_sign = np.sign(current)
        current_magnitude = np.absolute(current)
        actual_torque = np.interp(current_magnitude, self._current_table, self._torque_table)
        actual_torque = np.multiply(current_sign, actual_torque)
        actual_torque = np.multiply(self._strength_ratios, actual_torque)
        return actual_torque, observed_torque

class Minitaur(object):
    def __init__(self,
        pybullet_client,
        urdf_root = '',
        time_step = 0.01,
        action_repeat = 1,
        self_collision_enabled = False,
        motor_velocity_limit = np.inf,
        pd_control_enabled = False,
        accurate_motor_model_enabled = False,
        remove_default_joint_damping = False,
        motor_kp = 1.0,
        motor_kd = 0.02,
        pd_latency = 0.0,
        control_latency = 0.0,
        observation_noise_stdev = SENSOR_NOISE_STDDEV,
        torque_control_enabled = False,
        motor_overheat_protection = False,
        on_rack = False):
        print('INITIALIZE Minitaur')
        self.num_motors = 8
        self.num_legs = int(self.num_motors / 2)
        self._pybullet_client = pybullet_client
        self._action_repeat = action_repeat
        self._urdf_root = urdf_root
        self._self_collision_enabled = self_collision_enabled
        self._motor_velocity_limit = motor_velocity_limit
        self._pd_control_enabled = pd_control_enabled
        self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1]
        self._observed_motor_torques = np.zeros(self.num_motors)
        self._applied_motor_torques = np.zeros(self.num_motors)
        self._max_force = 3.5
        self._pd_latency = pd_latency
        self._control_latency = control_latency
        self._observation_noise_stdev = observation_noise_stdev
        self._accurate_motor_model_enabled = accurate_motor_model_enabled
        self._remove_default_joint_damping = remove_default_joint_damping
        self._observation_history = collections.deque(maxlen=100)
        self._control_observation = []
        self._chassis_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._foot_link_ids = []
        self._torque_control_enabled = torque_control_enabled
        self._motor_overheat_protection = motor_overheat_protection
        self._on_rack = on_rack
        if self._accurate_motor_model_enabled:
            self._kp = motor_kp
            self._kd = motor_kd
            self._motor_model = MotorModel(
                torque_control_enabled = self._torque_control_enabled,
                kp = self._kp,
                kd = self._kd)
        elif self._pd_control_enabled:
            self._kp = 8
            self._kd = 0.3
        else:
            self._kp = 1
            self._kd = 1
        self.time_step = time_step
        self._step_counter = 0
        print('-----Enter Minitaur.Reset1-----')
        self.Reset(reset_time = -1.0)
        print('-----Exit Minitaur.Reset1------')
        print('Minitaur INITIALIZED')

    def Reset(self, reload_urdf = True, default_motor_angles = None, reset_time = 3.0):
        print('reload_urdf?', reload_urdf, 'reset_time:', reset_time)
        if self._on_rack:
            init_position = INIT_RACK_POSITION
        else:
            init_position = INIT_POSITION
        if reload_urdf:
            if self._self_collision_enabled:
                self.quadruped = self._pybullet_client.loadURDF(
                    "%s/quadruped/minitaur.urdf" % self._urdf_root,
                    init_position,
                    useFixedBase=self._on_rack,
                    flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
            else:
                self.quadruped = self._pybullet_client.loadURDF(
                    "%s/quadruped/minitaur.urdf" % self._urdf_root,
                    init_position,
                    useFixedBase=self._on_rack)
            self._BuildJointNameToIdDict()
            self._BuildUrdfIds()
            if self._remove_default_joint_damping:
                self._RemoveDefaultJointDamping()
            self._BuildMotorIdList()
            self._RecordMassInfoFromURDF()
            self._RecordInertiaInfoFromURDF()
            self.ResetPose(add_constraint = True)
        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, init_position, INIT_ORIENTATION)
            print('Reset Position, Orientation')
            self._pybullet_client.resetBaseVelocity(self.quadruped, INIT_VELOCITY, INIT_ANGULAR_VELOCITY)
            print('Reset Velocity, Angular Velocity')
            self.ResetPose(add_constraint = False)
        self._overheat_counter = np.zeros(self.num_motors)
        self._motor_enabled_list = [True] * self.num_motors
        self._step_counter = 0
        self._observation_history.clear()
        self.ReceiveObservation()
        print('reload_urdf?', reload_urdf, 'reset_time:', reset_time)

    def Step(self, action, action_dot):
        for _ in range(self._action_repeat):
            self.ApplyAction(action, action_dot)
            self._pybullet_client.stepSimulation()
            self.ReceiveObservation()
            self._step_counter += 1

    def ApplyAction(self, motor_commands, velocity_commands, motor_kps = None, motor_kds = None):
        if motor_kps is None:
            motor_kps = np.full(8, self._kp)
        if motor_kds is None:
            motor_kds = np.full(8, self._kd)
        if self._accurate_motor_model_enabled or self._pd_control_enabled:
            q_true = self.GetTrueMotorAngles()
            qdot_true = self.GetTrueMotorVelocities()
            if self._accurate_motor_model_enabled:
                actual_torque, observed_torque = self._motor_model.convert_to_torque(motor_commands, q_true, velocity_commands, qdot_true, motor_kps, motor_kds)
                self._observed_motor_torques = observed_torque
                self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)
                for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list, self._applied_motor_torque, self._motor_enabled_list):
                    if motor_enabled:
                        self._SetMotorTorqueById(motor_id, motor_torque)
                    else:
                        self._SetMotorTorqueById(motor_id, 0)

    def _SetMotorTorqueById(self, motor_id, torque):
        self._pybullet_client.setJointMotorControl2(
            bodyIndex = self.quadruped,
            jointIndex = motor_id,
            controlMode = self._pybullet_client.TORQUE_CONTROL,
            force = torque)

    def GetTimeSinceReset(self):
        return self._step_counter * self.time_step

    def _BuildJointNameToIdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file."""
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._chassis_link_ids = [-1]
        # the self._leg_link_ids include both the upper and lower links of the leg.
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._foot_link_ids = []
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if _CHASSIS_NAME_PATTERN.match(joint_name):
                self._chassis_link_ids.append(joint_id)
            elif _MOTOR_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            elif _KNEE_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            else:
                self._leg_link_ids.append(joint_id)
        self._leg_link_ids.extend(self._foot_link_ids)
        self._chassis_link_ids.sort()
        self._motor_link_ids.sort()
        self._foot_link_ids.sort()
        self._leg_link_ids.sort()

    def _RemoveDefaultJointDamping(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(joint_info[0], -1, linearDamping = 0, angularDamping = 0)

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

    def _RecordMassInfoFromURDF(self):
        self._base_mass_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
        self._leg_masses_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
        for motor_id in self._motor_link_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])

    def _RecordInertiaInfoFromURDF(self):
        """Record the inertia of each body from URDF file."""
        self._link_urdf = []
        num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
        for body_id in range(-1, num_bodies):  # -1 is for the base link.
            inertia = self._pybullet_client.getDynamicsInfo(self.quadruped, body_id)[2]
            self._link_urdf.append(inertia)
        # We need to use id+1 to index self._link_urdf because it has the base
        # (index = -1) at the first element.
        self._base_inertia_urdf = [self._link_urdf[chassis_id + 1] for chassis_id in self._chassis_link_ids]
        self._leg_inertia_urdf = [self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids]
        self._leg_inertia_urdf.extend([self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

    def ResetPose(self, add_constraint):
        for i in range(self.num_legs):
            self._ResetPoseForLeg(i, add_constraint)

    def _ResetPoseForLeg(self, leg_id, add_constraint):
        knee_friction_force = 0
        leg_position = LEG_POSITION[leg_id]
        self._pybullet_client.resetJointState(self.quadruped,
                                              self._joint_name_to_id["motor_" + leg_position + "L_joint"],
                                              self._motor_direction[2 * leg_id] * IP_Joint_Position[4 * leg_id],
                                              targetVelocity = self._motor_direction[2 * leg_id] * IP_Joint_Velocity[4 * leg_id])
        self._pybullet_client.resetJointState(self.quadruped,
                                              self._joint_name_to_id["motor_" + leg_position + "R_joint"],
                                              self._motor_direction[2 * leg_id + 1] * IP_Joint_Position[4 * leg_id + 1],
                                              targetVelocity = self._motor_direction[2 * leg_id + 1] * IP_Joint_Velocity[4 * leg_id + 1])
        self._pybullet_client.resetJointState(self.quadruped,
                                              self._joint_name_to_id["knee_" + leg_position + "L_link"],
                                              self._motor_direction[2 * leg_id] * -(math.pi - IP_Joint_Position[4 * leg_id + 2]),
                                              targetVelocity = self._motor_direction[2 * leg_id] * -IP_Joint_Velocity[4 * leg_id + 2])
        self._pybullet_client.resetJointState(self.quadruped,
                                              self._joint_name_to_id["knee_" + leg_position + "R_link"],
                                              self._motor_direction[2 * leg_id + 1] * -(math.pi - IP_Joint_Position[4 * leg_id + 3]),
                                              targetVelocity = self._motor_direction[2 * leg_id + 1] * -IP_Joint_Velocity[4 * leg_id + 3])
        if add_constraint:
            self._pybullet_client.createConstraint(
                self.quadruped, self._joint_name_to_id["knee_" + leg_position + "R_link"],
                self.quadruped, self._joint_name_to_id["knee_" + leg_position + "L_link"],
                self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0], [0, 0.005, 0.2], [0, 0.01, 0.2])
        if self._accurate_motor_model_enabled or self._pd_control_enabled:
            self._pybullet_client.setJointMotorControl2(
                bodyIndex = self.quadruped, jointIndex = (self._joint_name_to_id["motor_" + leg_position + "L_joint"]),
                controlMode = self._pybullet_client.VELOCITY_CONTROL, targetVelocity = 0, force = knee_friction_force)
            self._pybullet_client.setJointMotorControl2(
                bodyIndex = self.quadruped, jointIndex = (self._joint_name_to_id["motor_" + leg_position + "R_joint"]),
                controlMode = self._pybullet_client.VELOCITY_CONTROL, targetVelocity = 0, force = knee_friction_force)
        else:
            self._pybullet_client.setJointMotorControl2(
                bodyIndex = self.quadruped, jointIndex = self._joint_name_to_id["motor_" + leg_position + "L_joint"],
                controlMode = self._pybullet_client.POSITION_CONTROL, targetPosition = self._motor_direction[2 * leg_id] * IP_Joint_Position[4 * leg_id],
                positionGain = self._kp, velocityGain = self._kd, force = self._max_force)
            self._pybullet_client.setJointMotorControl2(
                bodyIndex = self.quadruped, jointIndex = self._joint_name_to_id["motor_" + leg_position + "R_joint"],
                controlMode = self._pybullet_client.POSITION_CONTROL, targetPosition = self._motor_direction[2 * leg_id + 1] * IP_Joint_Position[4 * leg_id + 1],
                positionGain = self._kp, velocityGain = self._kd, force = self._max_force)
        self._pybullet_client.setJointMotorControl2(
            bodyIndex = self.quadruped, jointIndex = (self._joint_name_to_id["knee_" + leg_position + "L_link"]),
            controlMode = self._pybullet_client.VELOCITY_CONTROL, targetVelocity = 0, force = knee_friction_force)
        self._pybullet_client.setJointMotorControl2(
            bodyIndex = self.quadruped, jointIndex = (self._joint_name_to_id["knee_" + leg_position + "R_link"]),
            controlMode = self._pybullet_client.VELOCITY_CONTROL, targetVelocity = 0, force = knee_friction_force)

    def ReceiveObservation(self):
        self._observation_history.appendleft(self.GetTrueObservation())

    def GetTrueObservation(self):
        observation = []
        observation.extend(self.GetTrueMotorAngles())               #  0 -  7
        observation.extend(self.GetTrueMotorVelocities())           #  8 - 15
        observation.extend(self.GetTrueMotorTorques())              # 16 - 23
        observation.extend(self.GetTrueBaseOrientation())           # 24 - 27
        observation.extend(self.GetTrueBaseVelocity())              # 28 - 30
        observation.extend(self.GetTrueBaseRollPitchYawRate())      # 31 - 33
        return observation

    def GetTrueMotorAngles(self):
        motor_angles = [self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
                        for motor_id in self._motor_id_list]
        motor_angles = np.multiply(motor_angles, self._motor_direction)
        return motor_angles

    def GetTrueMotorVelocities(self):
        motor_velocities = [self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
                            for motor_id in self._motor_id_list]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetTrueMotorTorques(self):
        if self._accurate_motor_model_enabled or self._pd_control_enabled:
          return self._observed_motor_torques
        else:
          motor_torques = [self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
                           for motor_id in self._motor_id_list]
          motor_torques = np.multiply(motor_torques, self._motor_direction)
        return motor_torques

    def GetTrueBasePosition(self):
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
        return position

    def GetTrueBaseOrientation(self):
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
        return orientation

    def GetTrueBaseRollPitchYaw(self):
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetTrueBaseVelocity(self):
        vel = self._pybullet_client.getBaseVelocity(self.quadruped)
        return np.asarray([vel[0][0], vel[0][1], vel[0][2]])

    def GetTrueBaseRollPitchYawRate(self):
        vel = self._pybullet_client.getBaseVelocity(self.quadruped)
        return np.asarray([vel[1][0], vel[1][1], vel[1][2]])

    def Terminate(self):
        pass

class MinitaurGymEnv(gym.Env):
    def __init__(self,
        urdf_root = pybullet_data.getDataPath(), urdf_version = None,
        distance_weight = 1.0,
        energy_weight = 0.005,
        shake_weight = 0.0,
        drift_weight = 0.0,
        distance_limit = float("inf"),
        observation_noise_stdev = SENSOR_NOISE_STDDEV,
        self_collision_enabled = True,
        motor_velocity_limit = np.inf,
        pd_control_enabled = False,
        leg_model_enabled = True,
        accurate_motor_model_enabled = False,
        remove_default_joint_damping = False,
        motor_kp = 1.0, motor_kd = 0.02,
        control_latency = 0.0, pd_latency = 0.0,
        torque_control_enabled = False,
        motor_overheat_protection = False,
        hard_reset = True,
        on_rack = False,
        render = False,
        num_steps_to_log = 1000,
        action_repeat = 1,
        control_time_step = None,
        env_randomizer = None,
        forward_reward_cap = float("inf"),
        reflection = True,
        log_path = None):
        print('------------------------------INITIALIZE ENVIRONMENT------------------------------')
        self._log_path = log_path
        self.logging = minitaur_logging.MinitaurLogging(log_path)
        if control_time_step is not None:
            self.control_time_step = control_time_step          # 0.006
            self._action_repeat = action_repeat                 # 6
            self._time_step = control_time_step / action_repeat # 0.001
        else:
            if accurate_motor_model_enabled or pd_control_enabled:
                self._time_step = 0.002
                self._action_repeat = 5
            else:
                self._time_step = 0.01
                self._action_repeat = 1
            self.control_time_step = self._time_step * self._action_repeat
        self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
        self._urdf_root = urdf_root
        self._self_collision_enabled = self_collision_enabled
        self._motor_velocity_limit = motor_velocity_limit
        self._observation = []
        self._true_observation = []
        self._objectives = []
        self._objective_weights = [distance_weight, energy_weight, drift_weight, shake_weight]
        self._env_step_counter = 0
        self._num_steps_to_log = num_steps_to_log
        self._is_render = render
        self._last_base_position = [0, 0, 0]
        self._distance_weight = distance_weight
        self._energy_weight = energy_weight
        self._drift_weight = drift_weight
        self._shake_weight = shake_weight
        self._distance_limit = distance_limit
        self._observation_noise_stdev = observation_noise_stdev
        self._action_bound = 1
        self._pd_control_enabled = pd_control_enabled
        self._leg_model_enabled = leg_model_enabled
        self._accurate_motor_model_enabled = accurate_motor_model_enabled
        self._remove_default_joint_damping = remove_default_joint_damping
        self._motor_kp = motor_kp
        self._motor_kd = motor_kd
        self._torque_control_enabled = torque_control_enabled
        self._motor_overheat_protection = motor_overheat_protection
        self._on_rack = on_rack
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._forward_reward_cap = forward_reward_cap
        self._hard_reset = True
        self._last_frame_time = 0.0
        self._control_latency = control_latency
        self._pd_latency = pd_latency
        self._urdf_version = urdf_version
        self._ground_id = None
        self._reflection = reflection
        self._env_randomizers = convert_to_list(env_randomizer) if env_randomizer else []
        self._episode_proto = minitaur_logging_pb2.MinitaurEpisode()
        print('----------ENTER BulletClient----------')
        if self._is_render:
            self._pybullet_client = BC.BulletClient(connection_mode = pybullet.GUI)
            print('----------EXIT BulletClient----------')
        else:
            self._pybullet_client = BC.BulletClient()
            print('----------EXIT BulletClient----------')
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction = 0)
        self.seed()
        print('----------ENTER MinitaurGymEnv.reset1-----------')
        self.reset()
        print('----------EXIT MinitaurGymEnv.reset1------------')
        print('------------------------------ENVIRONMENT INITIALIZED------------------------------')

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, initial_motor_angles = None, reset_duration_test = 1.0):
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        if self._env_step_counter > 0:
            self.logging.save_episode(self._episode_proto)
        self._episode_proto = minitaur_logging_pb2.MinitaurEpisode()
        minitaur_logging.preallocate_episode_proto(self._episode_proto, self._num_steps_to_log)
        if self._hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
            if (self._reflection):
                self._pybullet_client.changeVisualShape(self._ground_id, -1, rgbaColor=[1, 1, 1, 0.8])
                self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, self._ground_id)
            self._pybullet_client.setGravity(0, 0, -10)
            acc_motor = self._accurate_motor_model_enabled
            motor_protect = self._motor_overheat_protection

        self.minitaur = Minitaur(
            pybullet_client = self._pybullet_client,
            action_repeat = self._action_repeat,
            urdf_root = self._urdf_root,
            time_step = self._time_step,
            self_collision_enabled = self._self_collision_enabled,
            motor_velocity_limit = self._motor_velocity_limit,
            pd_control_enabled = self._pd_control_enabled,
            accurate_motor_model_enabled = acc_motor,
            remove_default_joint_damping = self._remove_default_joint_damping,
            motor_kp = self._motor_kp, motor_kd = self._motor_kd,
            control_latency = self._control_latency, pd_latency = self._pd_latency,
            observation_noise_stdev = self._observation_noise_stdev,
            torque_control_enabled = self._torque_control_enabled,
            motor_overheat_protection = motor_protect,
            on_rack = self._on_rack)
        print('-----Enter Minitaur.Reset2-----')
        self.minitaur.Reset(reload_urdf = False, default_motor_angles = initial_motor_angles, reset_time = reset_duration_test)
        print('-----Exit Minitaur.Reset2------')
        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_env(self)

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction = 0)
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._objectives = []
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        return self._get_true_observation()

    def step(self, action, action_dot):
        self._last_base_position = self.minitaur.GetTrueBasePosition()
        if self._is_render:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.minitaur.GetTrueBasePosition()
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)
        self.minitaur.Step(action,action_dot)
        self._get_true_observation()
        done = self._termination()
        if self._log_path is not None:
            minitaur_logging.update_episode_proto(self._episode_proto, self.minitaur, action, self._env_step_counter)
        self._env_step_counter += 1
        if done:
            self.minitaur.Terminate()
        return np.array(self._get_true_observation())

    def _get_true_observation(self):
        """Get the observations of this environment.

        It includes the angles, velocities, torques and the orientation of the base.

        Returns:
          The observation list. observation[0:8] are motor angles. observation[8:16]
          are motor velocities, observation[16:24] are motor torques.
          observation[24:28] is base orientation, in quaternion form.
          observation[28:31] is base velocity
          observation[31:34] is base orientation, in roll-pitch-yaw form
        """
        observation = []
        observation.extend(self.minitaur.GetTrueMotorAngles().tolist())         #  0 -  7
        observation.extend(self.minitaur.GetTrueMotorVelocities().tolist())     #  8 - 15
        observation.extend(self.minitaur.GetTrueMotorTorques().tolist())        # 16 - 23
        observation.extend(list(self.minitaur.GetTrueBaseOrientation()))        # 24 - 27
        observation.extend(self.minitaur.GetTrueBaseVelocity().tolist())        # 28 - 30
        observation.extend(self.minitaur.GetTrueBaseRollPitchYaw().tolist())    # 31 - 33
        self._true_observation = observation
        return self._true_observation

    def _termination(self):
        position = self.minitaur.GetTrueBasePosition()
        distance = math.sqrt(position[0]**2 + position[1]**2)
        return self.is_fallen() or distance > self._distance_limit

    def is_fallen(self):
        orientation = self.minitaur.GetTrueBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.minitaur.GetTrueBasePosition()
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < 0.13)

class BoundingController(object):
    def __init__(self, robot):
        self._time = 0
        self._robot = robot

    def _motor_angles_to_leg_pose(self,motor_angles):
        leg_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            leg_pose[i] = 0.5 * (-1)**(i // 2) * (motor_angles[2 * i + 1] - motor_angles[2 * i])# swing
            leg_pose[NUM_LEGS + i] = 0.5 * (motor_angles[2 * i] + motor_angles[2 * i + 1])     # extension
        return leg_pose

    def _leg_pose_to_motor_angles(self,leg_pose):
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[2 * i] = leg_pose[NUM_LEGS + i] - (-1)**(i // 2) * leg_pose[i]
            motor_pose[2 * i + 1] = (leg_pose[NUM_LEGS + i] + (-1)**(i // 2) * leg_pose[i])
        return motor_pose

    def update(self, t, desired_motor_position, desired_velocity_x, desired_angle_pitch):
        self._time = t
        self._velocity = self._robot.GetTrueBaseVelocity()
        self._angle = self._robot.GetTrueBaseRollPitchYaw()
        desired_leg_pose = self._motor_angles_to_leg_pose(desired_motor_position)
        gait_time = math.fmod(self._time, 0.612)
        if gait_time <= 0.200:
            updated_leg_pose = self.FrontStanceController(desired_leg_pose, desired_angle_pitch)
        elif gait_time <= 0.300:
            updated_leg_pose = self.SwingController(desired_leg_pose, desired_velocity_x)
        elif gait_time <= 0.511:
            updated_leg_pose = self.BackStanceController(desired_leg_pose, desired_angle_pitch)
        else:
            updated_leg_pose = self.SwingController(desired_leg_pose, desired_velocity_x)
        updated_motor_position = self._leg_pose_to_motor_angles(updated_leg_pose)
        return updated_motor_position

    def FrontStanceController(self, desired_leg_pose, desired_angle_pitch):
        print('current pitch = ', self._angle[1])
        print('desired pitch = ', desired_angle_pitch)
        # if self._angle[1] > desired_angle_pitch, need to decrease pitch angle
        # by extending leg to give the front a push
        # by
        updated_leg_pose = [0] * 8
        if desired_leg_pose[0] < 0:
            # add a positive number to make the current leg pose less negative
            updated_leg_pose[0] = 1 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[0]
            updated_leg_pose[1] = desired_leg_pose[1]
            updated_leg_pose[2] = 1 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[2]
            updated_leg_pose[3] = desired_leg_pose[3]
            updated_leg_pose[4] = 1 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[4]
            updated_leg_pose[5] = desired_leg_pose[5]
            updated_leg_pose[6] = 1 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[6]
            updated_leg_pose[7] = desired_leg_pose[7]
        else:
            # add a negative number to make the current leg pose less positive
            updated_leg_pose[0] = -1 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[0]
            updated_leg_pose[1] = desired_leg_pose[1]
            updated_leg_pose[2] = -1 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[2]
            updated_leg_pose[3] = desired_leg_pose[3]
            updated_leg_pose[4] = 20 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[4]
            updated_leg_pose[5] = desired_leg_pose[5]
            updated_leg_pose[6] = 20 * (self._angle[1] - desired_angle_pitch) + desired_leg_pose[6]
            updated_leg_pose[7] = desired_leg_pose[7]

        print('desired front leg swing: ', desired_leg_pose[0])
        print('updated front leg swing: ', updated_leg_pose[0])
        print('desired front leg exten: ', desired_leg_pose[4])
        print('updated front leg swing: ', updated_leg_pose[4])
        return updated_leg_pose

    def BackStanceController(self, desired_leg_pose, desired_angle_pitch):
        updated_leg_pose = desired_leg_pose
        return updated_leg_pose

    def SwingController(self, desired_leg_pose, desired_velocity_x):
        updated_leg_pose = desired_leg_pose
        return updated_leg_pose

def main(argv):
    del argv
    try:
        env = MinitaurGymEnv(
            urdf_version = DEFAULT_URDF_VERSION,
            control_time_step = 0.006,
            action_repeat = 6,
            pd_latency = 0,
            control_latency = FLAGS.control_latency,
            motor_kp = FLAGS.motor_kp,
            motor_kd = FLAGS.motor_kd,
            remove_default_joint_damping = True,
            leg_model_enabled = False,
            render = True,
            on_rack = False,
            accurate_motor_model_enabled = True,
            log_path = FLAGS.log_path)
        print('------------------------------ENTER MinitaurGymEnv.reset2------------------------------')
        env.reset()
        print('-------------------------------Exit MinitaurGymEnv.reset2------------------------------')
        con = BoundingController(env.minitaur)
        print('START WALKING')
        num_iter = range(1000)
        for i in num_iter:
            input_idx = int(math.fmod(i,102))
            print('iteration number = ', i, 'input index = ', input_idx)

            t = env.minitaur.GetTimeSinceReset()
            desired_motor_position = Inputs_Joint_Position[input_idx]
            desired_motor_velocity = Inputs_Joint_Velocity[input_idx]
            desired_velocity_x = base_velocity_x_desire[input_idx]
            desired_angle_pitch = base_angle_pitch_desire[input_idx]

            #updated_motor_position = con.update(t, desired_motor_position, desired_velocity_x, desired_angle_pitch)
            updated_motor_position = desired_motor_position
            q_true = env.step(updated_motor_position, desired_motor_velocity)

            motor_flL_pos_desire, motor_flL_pos_actual, motor_flR_pos_desire, motor_flR_pos_actual = collect_data_front_left_position(desired_motor_position, q_true)
            motor_flL_vel_desire, motor_flL_vel_actual, motor_flR_vel_desire, motor_flR_vel_actual = collect_data_front_left_velocity(desired_motor_velocity, q_true)
            motor_blL_pos_desire, motor_blL_pos_actual, motor_blR_pos_desire, motor_blR_pos_actual = collect_data_back_left_position(desired_motor_position, q_true)
            motor_blL_vel_desire, motor_blL_vel_actual, motor_blR_vel_desire, motor_blR_vel_actual = collect_data_back_left_velocity(desired_motor_velocity, q_true)
            base_velocity_x_actual.append(q_true[28])
            base_angle_pitch_actual.append(q_true[32])

            input('Go To Next Iteration, Press Enter to continue...')
        print('DONE WALKING')
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(num_iter, motor_flL_pos_desire, 'b',
                 num_iter, motor_flL_pos_actual, 'r',
                 num_iter, motor_flR_pos_desire, 'b',
                 num_iter, motor_flR_pos_actual, 'r')
        plt.subplot(2,1,2)
        plt.plot(num_iter, motor_flL_vel_desire, 'b',
                 num_iter, motor_flL_vel_actual, 'r',
                 num_iter, motor_flR_vel_desire, 'b',
                 num_iter, motor_flR_vel_actual, 'r')
        plt.figure(2)
        plt.subplot(2,1,1)
        plt.plot(num_iter, motor_blL_pos_desire, 'b',
                 num_iter, motor_blL_pos_actual, 'r',
                 num_iter, motor_blR_pos_desire, 'b',
                 num_iter, motor_blR_pos_actual, 'r')
        plt.subplot(2,1,2)
        plt.plot(num_iter, motor_blL_vel_desire, 'b',
                 num_iter, motor_blL_vel_actual, 'r',
                 num_iter, motor_blR_vel_desire, 'b',
                 num_iter, motor_blR_vel_actual, 'r')
        plt.figure(3)
        plt.subplot(2,1,1)
        plt.plot(range(102), base_velocity_x_desire, 'b',
                 num_iter, base_velocity_x_actual,'r')
        plt.subplot(2,1,2)
        plt.plot(range(102), base_angle_pitch_desire, 'b',
                 num_iter, base_angle_pitch_actual, 'r')
        plt.draw()
    finally:
        env.close()
        plt.show()

def collect_data_front_left_position(action, q_true):
    motor_front_leftL_position_desire.append(action[0])
    motor_front_leftL_position_actual.append(q_true[0])
    motor_front_leftR_position_desire.append(action[1])
    motor_front_leftR_position_actual.append(q_true[1])
    return motor_front_leftL_position_desire, motor_front_leftL_position_actual, motor_front_leftR_position_desire, motor_front_leftR_position_actual

def collect_data_front_left_velocity(action_dot, q_true):
    motor_front_leftL_velocity_desire.append(action_dot[0])
    motor_front_leftL_velocity_actual.append(q_true[8])
    motor_front_leftR_velocity_desire.append(action_dot[1])
    motor_front_leftR_velocity_actual.append(q_true[9])
    return motor_front_leftL_velocity_desire, motor_front_leftL_velocity_actual, motor_front_leftR_velocity_desire, motor_front_leftR_velocity_actual

def collect_data_back_left_position(action, q_true):
    motor_back_leftL_position_desire.append(action[2])
    motor_back_leftL_position_actual.append(q_true[2])
    motor_back_leftR_position_desire.append(action[3])
    motor_back_leftR_position_actual.append(q_true[3])
    return motor_back_leftL_position_desire, motor_back_leftL_position_actual, motor_back_leftR_position_desire, motor_back_leftR_position_actual

def collect_data_back_left_velocity(action_dot, q_true):
    motor_back_leftL_velocity_desire.append(action_dot[2])
    motor_back_leftL_velocity_actual.append(q_true[10])
    motor_back_leftR_velocity_desire.append(action_dot[3])
    motor_back_leftR_velocity_actual.append(q_true[11])
    return motor_back_leftL_velocity_desire, motor_back_leftL_velocity_actual, motor_back_leftR_velocity_desire, motor_back_leftR_velocity_actual

if __name__ == "__main__":
    tf.compat.v1.app.run(main)
