"""A Raibert style controller for Minitaur."""

import collections
import math
import numpy as np

_NUM_MOTORS = 8
_NUM_LEGS = 4

_UPPER_LEG_LEN = 0.112
_LOWER_SHORT_LEG_LEN = 0.199
_LOWER_LONG_LEG_LEN = 0.2315

FRONT_LEG_PAIR = (0, 2)
BACK_LEG_PAIR = (1, 3)

class BehaviorParameters(collections.namedtuple("BehaviorParameters", [
        "stance_duration", "swing_duration",
        "desired_velocity_x", "desired_angle_pitch", "desired_position_z",
        "standing_height"])):
    __slots__ = ()

    def __new__(cls,
                stance_duration = 0.2, swing_duration = 0.4,
                desired_velocity_x = 0, desired_angle_pitch = 0, desired_position_z = 0.1914,
                standing_height = 0.1914):
        return super(BehaviorParameters, cls).__new__(cls,
            stance_duration, swing_duration,
            desired_velocity_x, desired_angle_pitch, desired_position_z,
            standing_height)

def motor_angles_to_leg_pose(motor_angles):
    leg_pose = np.zeros(_NUM_MOTORS)
    for i in range(_NUM_LEGS):
        leg_pose[i] = 0.5 * (-1)**(i // 2) * (motor_angles[2 * i + 1] - motor_angles[2 * i])
        leg_pose[_NUM_LEGS + i] = 0.5 * (motor_angles[2 * i] + motor_angles[2 * i + 1])
    return leg_pose

def leg_pose_to_motor_angles(leg_pose):
    motor_pose = np.zeros(_NUM_MOTORS)
    for i in range(_NUM_LEGS):
        motor_pose[2 * i] = leg_pose[_NUM_LEGS + i] - (-1)**(i // 2) * leg_pose[i]
        motor_pose[2 * i + 1] = (leg_pose[_NUM_LEGS + i] + (-1)**(i // 2) * leg_pose[i])
    return motor_pose

def foot_position_to_leg_pose(foot_position, current_pitch_angle):
    """The inverse kinematics."""
    """relative to the motor joint, base pitch angle considered"""
    l1 = _UPPER_LEG_LEN
    l2 = _LOWER_SHORT_LEG_LEN
    l3 = _LOWER_LONG_LEG_LEN

    x = foot_position[0]
    y = foot_position[1]
    assert (y < 0)
    hip_toe_sqr = x**2 + y**2
    cos_beta = (l1 * l1 + l3 * l3 - hip_toe_sqr) / (2 * l1 * l3)
    hip_ankle_sqr = l1 * l1 + l2 * l2 - 2 * l1 * l2 * cos_beta
    hip_ankle = math.sqrt(hip_ankle_sqr)
    cos_ext = -(l1 * l1 + hip_ankle_sqr - l2 * l2) / (2 * l1 * hip_ankle)
    ext = math.acos(cos_ext)

    hip_toe = math.sqrt(hip_toe_sqr)
    cos_theta = (hip_toe_sqr + hip_ankle_sqr - (l3 - l2)**2) / (2 * hip_ankle * hip_toe)

    assert cos_theta > 0
    theta = math.acos(cos_theta)
    sw = math.asin(x / hip_toe) - theta + current_pitch_angle
    return (-sw, ext)

def leg_pose_to_foot_position(leg_pose, current_pitch_angle):
    """The forward kinematics."""
    """relative to the motor joint, base pitch angle considered"""
    l1 = _UPPER_LEG_LEN
    l2 = _LOWER_SHORT_LEG_LEN
    l3 = _LOWER_LONG_LEG_LEN

    ext = leg_pose[1]
    alpha = math.asin(l1 * math.sin(ext) / l2)

    sw = leg_pose[0] + current_pitch_angle
    x = l3 * math.sin(alpha - sw) - l1 * math.sin(ext - sw)
    y = l3 * math.cos(alpha - sw) - l1 * math.cos(ext - sw)

    return (x, -y)

def extension_to_ankle_dist(ext):
    l1 = _UPPER_LEG_LEN
    l2 = _LOWER_SHORT_LEG_LEN
    l3 = _LOWER_LONG_LEG_LEN
    alpha = math.asin(l1 / l2 * math.sin(ext))
    return l2 * math.cos(alpha) - l1 * math.cos(ext)

def ankle_dist_to_extension(dist):
    l1 = _UPPER_LEG_LEN
    l2 = _LOWER_SHORT_LEG_LEN
    l3 = _LOWER_LONG_LEG_LEN
    cos_ext = -(l1**2 + dist**2 - l2**2) / (2 * l1 * dist)
    return math.acos(cos_ext)

def generate_swing_trajectory(phase, init_pose, end_pose):
    normalized_phase = min(phase * 3, 1)

    sw = (end_pose[0] - init_pose[0]) * normalized_phase + init_pose[0]
    ext = (end_pose[1] - init_pose[1]) * normalized_phase + init_pose[1]

    return (sw, ext)

def generate_stance_trajectory(phase, init_pose, end_pose):
    normalized_phase = phase

    sw = (end_pose[0] - init_pose[0]) * normalized_phase + init_pose[0]
    ext = (end_pose[1] - init_pose[1]) * normalized_phase + init_pose[1]

    return (sw, ext)

class RaibertSwingLegController(object):

    def __init__(self, speed_gain = -0.2, leg_trajectory_generator = generate_swing_trajectory):
        self._speed_gain = speed_gain
        self._leg_trajectory_generator = leg_trajectory_generator

    def get_action(self, raibert_controller, swing_set, swing_start_leg_pose, normalized_phase):
        current_speed = raibert_controller.estimate_base_velocity()
        current_angle = raibert_controller.estimate_base_angle()

        if swing_set[0] == 0:   # if front leg in swing
            target_leg_swing = -1 * current_angle
            target_leg_exten = raibert_controller.nominal_leg_extension
        else:                   # if back leg in swing
            target_leg_swing = -1 * current_angle - 0.0
            target_leg_exten = raibert_controller.nominal_leg_extension

        target_leg_pose = (target_leg_swing, target_leg_exten)
        print('swing set = ', swing_set[0], 'normalized_phase = ', normalized_phase)
        desired_leg_pose = self._leg_trajectory_generator(normalized_phase, swing_start_leg_pose, target_leg_pose)

        desired_motor_velocity = [-100, -100] # turn off D controller

        return desired_leg_pose, desired_motor_velocity

class RaibertStanceLegController(object):

    def __init__(self, speed_gain = 0.1, angle_gain = 1, leg_trajectory_generator = generate_stance_trajectory):
        self._speed_gain = speed_gain
        self._angle_gain = angle_gain
        self._leg_trajectory_generator = leg_trajectory_generator

    def get_action(self, raibert_controller, stance_set, stance_start_leg_pose, stance_action, stance_action_dot, index):
        current_speed = raibert_controller.estimate_base_velocity()
        current_angle = raibert_controller.estimate_base_angle()

        if index < len(stance_action):  # read stance leg pose data and motor velocity data regardless of which stance leg is action
            desired_leg_swing = stance_action[index][1] - current_angle
            desired_leg_exten = stance_action[index][2]
            desired_leg_pose = (desired_leg_swing, desired_leg_exten)
            desired_motor_velocity = stance_action_dot[index]
        else:                           # runs out of data
            if stance_set[0] == 0:      # if front leg in stance
                desired_leg_pose = (stance_action[-1][1], stance_action[-1][2])
            else:                       # if back leg in stance
                desired_leg_pose = (stance_action[-1][1] - current_angle, stance_action[-1][2])
            desired_motor_velocity = [-100, -100] # turn off D controller

        return desired_leg_pose, desired_motor_velocity

class MinitaurRaibertBoundingController(object):
    """A Raibert style controller for trotting gait."""
    def __init__(self, robot,
        behavior_parameters = BehaviorParameters(),
        swing_leg_controller = RaibertSwingLegController(),
        stance_leg_controller = RaibertStanceLegController()):
        self._time = 0
        self._robot = robot
        self._behavior_parameters = behavior_parameters
        self._swing_leg_controller = swing_leg_controller
        self._stance_leg_controller = stance_leg_controller

        nominal_leg_pose = foot_position_to_leg_pose((0, -self._behavior_parameters.standing_height), 0)
        self._nominal_leg_extension = nominal_leg_pose[1]

        self._phase_id = 1
        self._event_id = 1
        self._front = 1
        self._back = 0
        self._stance_start_front_leg_pose = self._get_average_leg_pose(FRONT_LEG_PAIR)
        self._swing_start_back_leg_pose   = self._get_average_leg_pose(BACK_LEG_PAIR)

        self._front_phase = -1
        self._back_phase  = -1

    def update(self, t):
        self._time = t
        front_left  = self._robot._pybullet_client.getClosestPoints(0, 1, 0.005, -1, 19)
        front_right = self._robot._pybullet_client.getClosestPoints(0, 1, 0.005, -1,  6)
        back_left   = self._robot._pybullet_client.getClosestPoints(0, 1, 0.005, -1, 22)
        back_right  = self._robot._pybullet_client.getClosestPoints(0, 1, 0.005, -1,  9)
        if (front_left and front_right) and not (back_left and back_right):
            if self._front == 0 and self._back == 0:
                print('Front Just Impacted')
                self._event_id = 1
                self._front_phase = -1
            elif self._front == 1 and self._back == 1:
                print('Back Just Lifted')
                self._event_id = 4
                self._back_phase = -1
            self._front = 1
            self._back = 0
            phase_id = 1
        elif (back_left and back_right) and not (front_left and front_right):
            if self._front == 0 and self._back == 0:
                print('Back Just Impacted')
                self._event_id = 3
                self._back_phase = -1
            elif self._front == 1 and self._back == 1:
                print('Front Just Lifted')
                self._event_id = 2
                self._front_phase = -1
            self._front = 0
            self._back = 1
            phase_id = 2
        elif not (back_left and back_right) and not (front_left and front_right):
            if self._front == 1 and self._back == 0:
                print('Front Just Lifted')
                self._event_id = 2
                self._front_phase = -1
            elif self._front == 0 and self._back == 1:
                print('Back Just Lifted')
                self._event_id = 4
                self._back_phase = -1
            self._front = 0
            self._back = 0
            phase_id = 3
        elif (front_left and front_right) and (back_left and back_right):
            if self._front == 1 and self._back == 0:
                print('Back Just Impacted')
                self._event_id = 3
                self._back_phase = -1
            elif self._front == 0 and self._back == 1:
                print('Front Just Impact')
                self._event_id = 1
                self._front_phase = -1
            self._front = 1
            self._back = 1
            phase_id = 4
        if phase_id is not self._phase_id:
            self._phase_id = phase_id
            # front impact
            if self._event_id == 1:
                self._stance_start_front_leg_pose = self._get_average_leg_pose(FRONT_LEG_PAIR)
            # front lift
            elif self._event_id == 2:
                self._swing_start_front_leg_pose = self._get_average_leg_pose(FRONT_LEG_PAIR)
            # back impact
            elif self._event_id == 3:
                self._stance_start_back_leg_pose = self._get_average_leg_pose(BACK_LEG_PAIR)
            # back lift
            elif self._event_id == 4:
                self._swing_start_back_leg_pose = self._get_average_leg_pose(BACK_LEG_PAIR)
        return self._phase_id, self._event_id

    def estimate_base_velocity(self):
        speed = self._robot.GetTrueBaseVelocity()
        return speed[0]

    def estimate_base_angle(self):
        angle = self._robot.GetTrueBaseRollPitchYaw()
        return angle[1]

    def _get_average_leg_pose(self, leg_indices):
        """Get the average leg pose."""
        current_leg_pose = motor_angles_to_leg_pose(self._robot.GetMotorAngles())
        # extract the swing leg pose from the current_leg_pose
        leg_pose = []
        for index in leg_indices:
            leg_pose.append([current_leg_pose[index], current_leg_pose[index + _NUM_LEGS]])

        leg_pose = np.array(leg_pose)
        return np.mean(leg_pose, axis=0)

    def get_action(self, front_stance_action, back_stance_action, front_stance_action_dot, back_stance_action_dot):
        self._front_phase += 1
        self._back_phase  += 1

        # Front Stance
        if self._phase_id == 1:
            front_leg_pose, front_motor_velocity = self._stance_leg_controller.get_action(self, FRONT_LEG_PAIR, self._stance_start_front_leg_pose, front_stance_action, front_stance_action_dot, self._front_phase)
            back_leg_pose,  back_motor_velocity  = self._swing_leg_controller. get_action(self, BACK_LEG_PAIR,  self._swing_start_back_leg_pose,   self._back_phase / 68)
        # Back Stance
        elif self._phase_id == 2:
            front_leg_pose, front_motor_velocity = self._swing_leg_controller. get_action(self, FRONT_LEG_PAIR, self._swing_start_front_leg_pose,  self._front_phase / 66)
            back_leg_pose,  back_motor_velocity  = self._stance_leg_controller.get_action(self, BACK_LEG_PAIR,  self._stance_start_back_leg_pose,  back_stance_action, back_stance_action_dot, self._back_phase)
        # Flight
        elif self._phase_id == 3:
            front_leg_pose, front_motor_velocity = self._swing_leg_controller. get_action(self, FRONT_LEG_PAIR, self._swing_start_front_leg_pose,  self._front_phase / 66)
            back_leg_pose,  back_motor_velocity  = self._swing_leg_controller. get_action(self, BACK_LEG_PAIR,  self._swing_start_back_leg_pose,   self._back_phase / 68)
        # Stance
        elif self._phase_id == 4:
            front_leg_pose, front_motor_velocity = self._stance_leg_controller.get_action(self, FRONT_LEG_PAIR, self._stance_start_front_leg_pose, front_stance_action, front_stance_action_dot, self._front_phase)
            back_leg_pose,  back_motor_velocity  = self._stance_leg_controller.get_action(self, BACK_LEG_PAIR,  self._stance_start_back_leg_pose,  back_stance_action,  back_stance_action_dot,  self._back_phase )

        leg_pose = [0] * _NUM_MOTORS
        motor_velocity = [0] * _NUM_MOTORS

        for i in FRONT_LEG_PAIR:    # 0, 2
            leg_pose[i] = front_leg_pose[0]                         # front_swing
            leg_pose[i + _NUM_LEGS] = front_leg_pose[1]             # front_exten
            if i == 0:              # front_left_leg
                motor_velocity[i * 2] = front_motor_velocity[0]     # motor 7
                motor_velocity[i * 2 + 1] = front_motor_velocity[1] # motor 8
            else:                   # front_right_leg
                motor_velocity[i * 2] = front_motor_velocity[1]     # motor 8
                motor_velocity[i * 2 + 1] = front_motor_velocity[0] # motor 7

        for i in BACK_LEG_PAIR:     # 1, 3
            leg_pose[i] = back_leg_pose[0]                          # back_swing
            leg_pose[i + _NUM_LEGS] = back_leg_pose[1]              # back_exten
            if i == 1:              # back_left_leg
                motor_velocity[i * 2] = back_motor_velocity[0]      # motor 11
                motor_velocity[i * 2 + 1] = back_motor_velocity[1]  # motor 12
            else:                   # back_right_leg
                motor_velocity[i * 2] = back_motor_velocity[1]      # motor 12
                motor_velocity[i * 2 + 1] = back_motor_velocity[0]  # motor 11

        # leg pose index: 0             1               2               3               4               5               6               7
        # leg pose value: front_swing   back_swing      front_swing     back_swing      front_exten     back_exten      front_exten     back_swing

        # mot velo index: 0             1               2               3               4               5               6               7
        # mot velo value: motor 7       motor 8         motor 11        motor 12        motor 8         motor 7         motor 12        motor 11
        return leg_pose_to_motor_angles(leg_pose), motor_velocity


    @property
    def behavior_parameters(self):
        return self._behavior_parameters

    @behavior_parameters.setter
    def behavior_parameters(self, behavior_parameters):
        self._behavior_parameters = behavior_parameters

    @property
    def nominal_leg_extension(self):
      return self._nominal_leg_extension
