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
        "front_stance_duration", "front_swing_duration",
        "back_stance_duration", "back_swing_duration",
        "desired_velocity_x", "desired_angle_pitch", "desired_position_z"])):
    __slots__ = ()

    def __new__(cls,
                front_stance_duration = 0.2, front_swing_duration = 0.411,
                back_stance_duration = 0.211, back_swing_duration = 0.4,
                desired_velocity_x = 0.7, desired_angle_pitch = 0, desired_position_z = 0.25):
        return super(BehaviorParameters, cls).__new__(cls,
            front_stance_duration, front_swing_duration,
            back_stance_duration, back_swing_duration,
            desired_velocity_x, desired_angle_pitch, desired_position_z)

def motor_angles_to_leg_pose(motor_angles):
    leg_pose = np.zeros(_NUM_MOTORS)
    for i in range(_NUM_LEGS):
        leg_pose[i] = 0.5 * (-1)**(i // 2) * (motor_angles[2 * i + 1] - motor_angles[2 * i])
        leg_pose[_NUM_LEGS + i] = 0.5 * (motor_angles[2 * i] + motor_angles[2 * i + 1])
    return leg_pose

def leg_pose_to_motor_angles(leg_pose):
    print(leg_pose)
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
    normalized_phase = min(phase * 5, 1)

    # For swing, we use a linear interpolation:
    sw = (end_pose[0] - init_pose[0]) * normalized_phase + init_pose[0]

    # For extension, we can fit a second order polynomial:
    min_ext = (init_pose[1] + end_pose[1]) / 2 - 1.2
    min_ext = max(min_ext, 0.5)
    phi = 0.7

    min_delta = extension_to_ankle_dist(min_ext)
    init_delta = extension_to_ankle_dist(init_pose[1])
    end_delta = extension_to_ankle_dist(end_pose[1])

    # The polynomial is: a * phi^2 + b * phi + c
    delta_1 = min_delta - init_delta
    delta_2 = end_delta - init_delta
    delta_p = phi * phi - phi

    a = (delta_1 - phi * delta_2) / delta_p

    b = (phi * phi * delta_2 - delta_1) / delta_p

    delta = (a * normalized_phase * normalized_phase + b * normalized_phase + init_delta)

    l1 = _UPPER_LEG_LEN
    l2 = _LOWER_SHORT_LEG_LEN

    delta = min(max(delta, l2 - l1 + 0.01), l2 + l1 - 0.01)

    ext = ankle_dist_to_extension(delta)
    ext = end_pose[1]
    return (sw, ext)

def generate_stance_trajectory(phase, init_pose, end_pose):
    normalized_phase = phase
    sw = (end_pose[0] - init_pose[0]) * normalized_phase + init_pose[0]
    ext = (end_pose[1] - init_pose[1]) * normalized_phase + init_pose[1]
    return (sw, ext)

class RaibertSwingLegController(object):
    def __init__(self, speed_gain = -0.2, leg_extension_clearance = 0.3, leg_trajectory_generator = generate_swing_trajectory):
        self._speed_gain = speed_gain
        self._leg_extension_clearance = leg_extension_clearance
        self._leg_trajectory_generator = leg_trajectory_generator

    def get_action(self, raibert_controller, swing_set, swing_start_leg_pose, phase):
        print('Swing Controller')
        leg_pose_set = []
        for i in swing_set:
            target_leg_pose = (-0.5, 2)
            desired_leg_pose = self._leg_trajectory_generator(phase, swing_start_leg_pose, target_leg_pose)
            print('desired swing leg set = ', desired_leg_pose)
            leg_pose_set.append(desired_leg_pose)
        return leg_pose_set

class RaibertStanceLegController(object):
    def __init__(self, speed_gain = 0.1, angle_gain = 1, leg_trajectory_generator = generate_stance_trajectory):
        self._speed_gain = speed_gain
        self._angle_gain = angle_gain
        self._leg_trajectory_generator = leg_trajectory_generator

    def get_action(self, raibert_controller, stance_set, stance_start_leg_pose, phase):
        print('Stance Controller')
        leg_pose_set = []
        for i in stance_set:
            desired_forward_speed = raibert_controller.behavior_parameters.desired_velocity_x
            desired_tilting_angle = raibert_controller.behavior_parameters.desired_angle_pitch
            desired_height = raibert_controller.behavior_parameters.desired_position_z
            y = desired_height - abs(0.2 * math.sin(desired_tilting_angle))
            x = self.stance_x - 0.001 * desired_forward_speed
            target_leg_pose = foot_position_to_leg_pose([x,-y], desired_tilting_angle)
            desired_leg_pose = self._leg_trajectory_generator(phase, stance_start_leg_pose, target_leg_pose)
            leg_pose_set.append(desired_leg_pose)

        self.stance_x = x
        return leg_pose_set


class MinitaurRaibertBoundingController(object):
    def __init__(self, robot,
        behavior_parameters = BehaviorParameters(),
        swing_leg_controller = RaibertSwingLegController(),
        stance_leg_controller = RaibertStanceLegController()):
        self._time = 0
        self._robot = robot
        self._behavior_parameters = behavior_parameters

        self._swing_leg_controller = swing_leg_controller
        self._stance_leg_controller = stance_leg_controller

        self._phase_id = 1
        # stance phase for each leg is happening at different time interval, no need to differentiate them
        self._stance_start_leg_pose = self._get_average_leg_pose(FRONT_LEG_PAIR)
        # swing phase for each leg could happen at the same time interval, need to differentiate them
        self._swing_start_back_leg_pose = self._get_average_leg_pose(BACK_LEG_PAIR)

        # for stance controller, need to know the initial x position of the foot with resptive to the hip
        current_pitch_angle = self.estimate_base_angle()
        inital_stance_foot_position = leg_pose_to_foot_position(self._stance_start_leg_pose, current_pitch_angle)
        self._stance_leg_controller.stance_x = inital_stance_foot_position[0]

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

    def update(self, t):
        print('time = ', t)
        self._time = t
        """Switch the set of swing/stance legs based on timing."""
        swing_stance_phase = math.fmod(self._time, 0.610)
        current_pitch_angle = self.estimate_base_angle()
        if swing_stance_phase < 0.2:
            stance_set, swing_set = FRONT_LEG_PAIR, BACK_LEG_PAIR
            phase_id = 1
        elif swing_stance_phase < 0.3:
            stance_set, swing_set = (), (0,1,2,3)
            phase_id = 2
        elif swing_stance_phase < 0.511:
            stance_set, swing_set = BACK_LEG_PAIR, FRONT_LEG_PAIR
            phase_id = 3
        else:
            stance_set, swing_set = (), (0,1,2,3)
            phase_id = 4
        if phase_id is not self._phase_id:
            self._phase_id = phase_id
            print('Enter into new phase ', self._phase_id)
            # if stance phase, record the respective leg start pose
            if self._phase_id == 1 or self._phase_id == 3:
                self._stance_set = stance_set
                self._stance_start_leg_pose = self._get_average_leg_pose(self._stance_set)
                inital_stance_foot_position = leg_pose_to_foot_position(self._stance_start_leg_pose, current_pitch_angle)
                self._stance_leg_controller.stance_x = inital_stance_foot_position[0]
            # if flight1 phase, record front leg start pose
            elif self._phase_id == 2:
                self._swing_start_front_leg_pose = self._get_average_leg_pose(FRONT_LEG_PAIR)
            # if flight2 phase, record back leg start pose
            else:
                self._swing_start_back_leg_pose = self._get_average_leg_pose(BACK_LEG_PAIR)

    def get_swing_leg_action(self, swing_set, swing_start_leg_pose, phase):
        return self._swing_leg_controller.get_action(self, swing_set, swing_start_leg_pose, phase)

    def get_stance_leg_action(self, stance_set, stance_start_leg_pose, phase):
        return self._stance_leg_controller.get_action(self, stance_set, stance_start_leg_pose, phase)

    def get_action(self):
        leg_pose = [0] * _NUM_MOTORS

        if self._phase_id == 1:
            front_stance_phase = math.fmod(self._time, 0.61) / self._behavior_parameters.front_stance_duration
            back_swing_phase = (math.fmod(self._time, 0.61) + 0.1) / self._behavior_parameters.back_swing_duration
            front_leg_pose = self.get_stance_leg_action(FRONT_LEG_PAIR, self._stance_start_leg_pose, front_stance_phase)
            back_leg_pose = self.get_swing_leg_action(BACK_LEG_PAIR, self._swing_start_back_leg_pose, back_swing_phase)
        elif self._phase_id == 2:
            front_swing_phase = (math.fmod(self._time, 0.61) - 0.2) / self._behavior_parameters.front_swing_duration
            back_swing_phase = (math.fmod(self._time, 0.61) + 0.1) / self._behavior_parameters.back_swing_duration
            front_leg_pose = self.get_swing_leg_action(FRONT_LEG_PAIR, self._swing_start_front_leg_pose, front_swing_phase)
            back_leg_pose = self.get_swing_leg_action(BACK_LEG_PAIR, self._swing_start_back_leg_pose, back_swing_phase)
        elif self._phase_id == 3:
            front_swing_phase = (math.fmod(self._time, 0.61) - 0.2) / self._behavior_parameters.front_swing_duration
            back_stance_phase = (math.fmod(self._time, 0.61) - 0.3) / self._behavior_parameters.back_stance_duration
            front_leg_pose = self.get_swing_leg_action(FRONT_LEG_PAIR, self._swing_start_front_leg_pose, front_swing_phase)
            back_leg_pose = self.get_stance_leg_action(BACK_LEG_PAIR, self._stance_start_leg_pose, back_stance_phase)
        else:
            front_swing_phase = (math.fmod(self._time, 0.61) - 0.2) / self._behavior_parameters.front_swing_duration
            back_swing_phase = (math.fmod(self._time, 0.61) - 0.511) / self._behavior_parameters.back_swing_duration
            front_leg_pose = self.get_swing_leg_action(FRONT_LEG_PAIR, self._swing_start_front_leg_pose, front_swing_phase)
            back_leg_pose = self.get_swing_leg_action(BACK_LEG_PAIR, self._swing_start_back_leg_pose, back_swing_phase)

        j = 0
        for i in FRONT_LEG_PAIR:
            leg_pose[i] = front_leg_pose[j][0]
            leg_pose[i + _NUM_LEGS] = front_leg_pose[j][1]
            j += 1

        j = 0
        for i in BACK_LEG_PAIR:
            leg_pose[i] = back_leg_pose[j][0]
            leg_pose[i + _NUM_LEGS] = back_leg_pose[j][1]
            j += 1

        return leg_pose_to_motor_angles(leg_pose), leg_pose