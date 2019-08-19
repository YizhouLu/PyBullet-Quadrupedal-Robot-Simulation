from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

for _ in range(1):
    import os, inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
    os.sys.path.insert(0, parentdir)

    import tensorflow as tf, math, time, numpy as np, collections, copy, re, gym, pybullet
    from pybullet_envs.minitaur.envs import minitaur_bounding_controller_v3 as minitaur_bounding_controller
    from gym import spaces
    from gym.utils import seeding
    from pybullet_envs.minitaur.envs import bullet_client as BC, minitaur_logging as ML, minitaur_logging_pb2 as ML2
    import pybullet_data
    from pkg_resources import parse_version
    import scipy.io as sio
    import matplotlib.pyplot as plt
    # MOTOR
    for _ in range(1):
        VOLTAGE_CLIPPING = 50
        OBSERVED_TORQUE_LIMIT = 5.7
        MOTOR_VOLTAGE = 16.0
        MOTOR_RESISTANCE = 0.186
        MOTOR_TORQUE_CONSTANT = 0.0954
        MOTOR_VISCOUS_DAMPING = 0
        MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING + MOTOR_TORQUE_CONSTANT)
        NUM_MOTORS = 8
    # MINITAUR
    for _ in range(1):
        mat1 = sio.loadmat('vel_07_InitialCondition_to_VS_0006.mat')
        sorted(mat1.keys())
        IP_Torso_Position = mat1['init_pos']
        IP_Torso_Rotation = mat1['init_rot']
        IP_Torso_Velocity = mat1['init_vel']
        IP_Torso_AngularV = mat1['init_avel']
        IP_Joint_Position = mat1['init_joint']
        IP_Joint_Velocity = mat1['init_jvel']

        INIT_POSITION = [IP_Torso_Position[0], IP_Torso_Position[1], IP_Torso_Position[2]]
        INIT_RACK_POSITION = [0, 0, 1]
        INIT_ORIENTATION = [IP_Torso_Rotation[0], IP_Torso_Rotation[1], IP_Torso_Rotation[2], IP_Torso_Rotation[3]]
        INIT_VELOCITY = [IP_Torso_Velocity[0], IP_Torso_Velocity[1], IP_Torso_Velocity[2]]
        INIT_ANGULAR_VELOCITY = [IP_Torso_AngularV[0],IP_Torso_AngularV[1],IP_Torso_AngularV[2]]
        #INIT_ANGULAR_VELOCITY = [0, 0, 0]
        """
        INIT_POSITION = [0, 0, 0.2]
        INIT_RACK_POSITION = [0, 0, 1]
        INIT_ORIENTATION = [0, 0, 0, 1]
        INIT_VELOCITY = [0, 0, 0]
        INIT_ANGULAR_VELOCITY = [0, 0, 0]
        """
        OVERHEAT_SHUTDOWN_TORQUE = 2.45
        OVERHEAT_SHUTDOWN_TIME = 1.0
        LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]
        MOTOR_NAMES = [
            "motor_front_leftL_joint", "motor_front_leftR_joint", "motor_back_leftL_joint",
            "motor_back_leftR_joint", "motor_front_rightL_joint", "motor_front_rightR_joint",
            "motor_back_rightL_joint", "motor_back_rightR_joint"
        ]
        _CHASSIS_NAME_PATTERN = re.compile(r"chassis\D*center")
        _MOTOR_NAME_PATTERN = re.compile(r"motor\D*joint")
        _KNEE_NAME_PATTERN = re.compile(r"knee\D*")
        SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)
        TWO_PI = 2 * math.pi
    # ENVIRONMENT
    for _ in range(1):
        NUM_MOTORS = 8
        MOTOR_ANGLE_OBSERVATION_INDEX = 0
        MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
        MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
        BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
        ACTION_EPS = 0.01
        OBSERVATION_EPS = 0.01
        RENDER_HEIGHT = 360
        RENDER_WIDTH = 480
        NUM_SIMULATION_ITERATION_STEPS = 300
    # MAIN
    for _ in range(1):
        mat2 = sio.loadmat('vel_07_States_to_VS_0006.mat')
        sorted(mat2.keys())
        base_velocity_x_desire = mat2['base_velocity_x']
        base_angle_pitch_desire = mat2['base_angle_pitch']
        base_position_z_desire = mat2['base_position_z']

        mat3 = sio.loadmat('vel_07_Inputs_to_VS_0006.mat')
        sorted(mat1.keys())
        Inputs_Joint_Position = mat3['int_joint_position']
        Inputs_Joint_Velocity = mat3['int_joint_velocity']

        mat4 = sio.loadmat('vel_07_Inputs_to_VS_0006_FrontStance')
        sorted(mat4.keys())
        Inputs_Joint_Position_Front_Stance = mat4['int_front_leg_pose']
        Inputs_Joint_Velocity_Front_Stance = mat4['int_front_leg_velocity']

        mat5 = sio.loadmat('vel_07_Inputs_to_VS_0006_BackStance')
        sorted(mat5.keys())
        Inputs_Joint_Position_Back_Stance = mat5['int_back_leg_pose']
        Inputs_Joint_Velocity_Back_Stance = mat5['int_back_leg_velocity']

        flags = tf.app.flags
        FLAGS = tf.app.flags.FLAGS

        flags.DEFINE_float("motor_kp", 5.0, "The position gain of the motor.")
        flags.DEFINE_float("motor_kd", 0.1, "The speed gain of the motor.")
        flags.DEFINE_float("control_latency", 0.02, "The latency between sensor measurement and action"
                           " execution the robot.")
        flags.DEFINE_string("log_path", None, "The directory to write the log file.")

        front_left_leg_swing_actual = []
        front_left_leg_exten_actual = []
        front_left_leg_swing_desire = []
        front_left_leg_exten_desire = []

        front_phase_all = []
        back_phase_all = []

def MapToMinusPiToPi(angles):
    """Maps a list of angles to [-pi, pi].

    Args:
        angles: A list of angles in rad.
    Returns:
        A list of angle mapped to [-pi, pi].
    """
    mapped_angles = copy.deepcopy(angles)
    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], TWO_PI)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= TWO_PI
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += TWO_PI
    return mapped_angles

def convert_to_list(obj):
    try:
        iter(obj)
        return obj
    except TypeError:
        return [obj]

def velocity(iter):
    return base_velocity_x_desire[iter]

class MotorModel(object):
  """The accurate motor model, which is based on the physics of DC motors.

  The motor model support two types of control: position control and torque
  control. In position control mode, a desired motor angle is specified, and a
  torque is computed based on the internal motor model. When the torque control
  is specified, a pwm signal in the range of [-1.0, 1.0] is converted to the
  torque.

  The internal motor model takes the following factors into consideration:
  pd gains, viscous friction, back-EMF voltage and current-torque profile.
  """

  def __init__(self, torque_control_enabled=False, kp=1.2, kd=0):
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

  def set_strength_ratios(self, ratios):
    """Set the strength of each motors relative to the default value.

    Args:
      ratios: The relative strength of motor output. A numpy array ranging from
        0.0 to 1.0.
    """
    self._strength_ratios = np.array(ratios)

  def set_motor_gains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
    self._kp = kp
    self._kd = kd

  def set_voltage(self, voltage):
    self._voltage = voltage

  def get_voltage(self):
    return self._voltage

  def set_viscous_damping(self, viscous_damping):
    self._viscous_damping = viscous_damping

  def get_viscous_dampling(self):
    return self._viscous_damping

  def convert_to_torque(self, angle_commands, true_motor_angle, velocity_commands, true_motor_velocity, kp=None, kd=None):
    """Convert the commands (position control or torque control) to torque.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      motor_angle: The motor angle observed at the current time step. It is
        actually the true motor angle observed a few milliseconds ago (pd
        latency).
      motor_velocity: The motor velocity observed at the current time step, it
        is actually the true motor velocity a few milliseconds ago (pd latency).
      true_motor_velocity: The true motor velocity. The true velocity is used
        to compute back EMF voltage and viscous damping.
      kp: Proportional gains for the motors' PD controllers. If not provided, it
        uses the default kp of the minitaur for all the motors.
      kd: Derivative gains for the motors' PD controllers. If not provided, it
        uses the default kp of the minitaur for all the motors.

    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    pwm = [0] * 8
    if self._torque_control_enabled:
        pwm = motor_commands
    else:
        print('input front swing = ', (angle_commands[1]-angle_commands[0])/2)
        print('input front exten = ', (angle_commands[1]+angle_commands[0])/2)
        print('input back swing = ', (angle_commands[3]-angle_commands[2])/2)
        print('input back exten = ', (angle_commands[3]+angle_commands[2])/2)
        print('input velocity = ', velocity_commands)
        for i in range(8):
            if  velocity_commands[i] == -100:
                pwm[i] = self._kp * (angle_commands[i] - true_motor_angle[i])
                #print('P control')
            else:
                pwm[i] = self._kp * (angle_commands[i] - true_motor_angle[i]) + self._kd * (velocity_commands[i] - true_motor_velocity[i])
                #print('PD control')
    pwm = np.clip(pwm, -1.0, 1.0)
    return self._convert_to_torque_from_pwm(pwm, true_motor_velocity)

  def _convert_to_torque_from_pwm(self, pwm, true_motor_velocity):
    """Convert the pwm signal to torque.

    Args:
      pwm: The pulse width modulation.
      true_motor_velocity: The true motor velocity at the current moment. It is
        used to compute the back EMF voltage and the viscous damping.
    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    observed_torque = np.clip(
        self._torque_constant * (np.asarray(pwm) * self._voltage / self._resistance),
        -OBSERVED_TORQUE_LIMIT, OBSERVED_TORQUE_LIMIT)

    # Net voltage is clipped at 50V by diodes on the motor controller.
    voltage_net = np.clip(
        np.asarray(pwm) * self._voltage -
        (self._torque_constant + self._viscous_damping) * np.asarray(true_motor_velocity),
        -VOLTAGE_CLIPPING, VOLTAGE_CLIPPING)
    current = voltage_net / self._resistance
    current_sign = np.sign(current)
    current_magnitude = np.absolute(current)
    # Saturate torque based on empirical current relation.
    actual_torque = np.interp(current_magnitude, self._current_table, self._torque_table)
    actual_torque = np.multiply(current_sign, actual_torque)
    actual_torque = np.multiply(self._strength_ratios, actual_torque)
    return actual_torque, observed_torque

class Minitaur(object):
  """The minitaur class that simulates a quadruped robot from Ghost Robotics.

  """

  def __init__(self,
               pybullet_client,
               urdf_root="",
               time_step=0.01,
               action_repeat=1,
               self_collision_enabled=False,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               accurate_motor_model_enabled=False,
               remove_default_joint_damping=False,
               motor_kp=1.0,
               motor_kd=0.02,
               pd_latency=0.0,
               control_latency=0.0,
               observation_noise_stdev=SENSOR_NOISE_STDDEV,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               on_rack=False):
    """Constructs a minitaur and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      urdf_root: The path to the urdf folder.
      time_step: The time step of the simulation.
      action_repeat: The number of ApplyAction() for each control step.
      self_collision_enabled: Whether to enable self collision.
      motor_velocity_limit: The upper limit of the motor velocity.
      pd_control_enabled: Whether to use PD control for the motors.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      remove_default_joint_damping: Whether to remove the default joint damping.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      pd_latency: The latency of the observations (in seconds) used to calculate
        PD control. On the real hardware, it is the latency between the
        microcontroller and the motor controller.
      control_latency: The latency of the observations (in second) used to
        calculate action. On the real hardware, it is the latency from the motor
        controller, the microcontroller to the host (Nvidia TX2).
      observation_noise_stdev: The standard deviation of a Gaussian noise model
        for the sensor. It should be an array for separate sensors in the
        following order [motor_angle, motor_velocity, motor_torque,
        base_roll_pitch_yaw, base_angular_velocity]
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
    """
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
      self._motor_model = MotorModel(torque_control_enabled=self._torque_control_enabled,
                                           kp=self._kp,
                                           kd=self._kd)
    elif self._pd_control_enabled:
      self._kp = 8
      self._kd = 0.3
    else:
      self._kp = 1
      self._kd = 1
    self.time_step = time_step
    self._step_counter = 0
    # reset_time=-1.0 means skipping the reset motion.
    # See Reset for more details.
    self.Reset(reset_time=-1.0)

  def GetTimeSinceReset(self):
    return self._step_counter * self.time_step

  def Step(self, action, action_dot):
    for _ in range(self._action_repeat):
      self.ApplyAction(action, action_dot)
      self._pybullet_client.stepSimulation()
      self.ReceiveObservation()
      self._step_counter += 1

  def Terminate(self):
    pass

  def _RecordMassInfoFromURDF(self):
    self._base_mass_urdf = []
    for chassis_id in self._chassis_link_ids:
      self._base_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
    self._leg_masses_urdf = []
    for leg_id in self._leg_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
    for motor_id in self._motor_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])

  def _RecordInertiaInfoFromURDF(self):
    """Record the inertia of each body from URDF file."""
    self._link_urdf = []
    num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
    for body_id in range(-1, num_bodies):  # -1 is for the base link.
      inertia = self._pybullet_client.getDynamicsInfo(self.quadruped, body_id)[2]
      self._link_urdf.append(inertia)
    # We need to use id+1 to index self._link_urdf because it has the base
    # (index = -1) at the first element.
    self._base_inertia_urdf = [
        self._link_urdf[chassis_id + 1] for chassis_id in self._chassis_link_ids
    ]
    self._leg_inertia_urdf = [self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids]
    self._leg_inertia_urdf.extend(
        [self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

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
      self._pybullet_client.changeDynamics(joint_info[0], -1, linearDamping=0, angularDamping=0)

  def _BuildMotorIdList(self):
    self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

  def IsObservationValid(self):
    """Whether the observation is valid for the current time step.

    In simulation, observations are always valid. In real hardware, it may not
    be valid from time to time when communication error happens between the
    Nvidia TX2 and the microcontroller.

    Returns:
      Whether the observation is valid for the current time step.
    """
    return True

  def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
      default_motor_angles: The default motor angles. If it is None, minitaur
        will hold a default pose (motor angle math.pi / 2) for 100 steps. In
        torque control mode, the phase of holding the default pose is skipped.
      reset_time: The duration (in seconds) to hold the default motor angles. If
        reset_time <= 0 or in torque control mode, the phase of holding the
        default pose is skipped.
    """
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
        self.quadruped = self._pybullet_client.loadURDF("%s/quadruped/minitaur.urdf" %
                                                        self._urdf_root,
                                                        init_position,
                                                        useFixedBase=self._on_rack)
      self._BuildJointNameToIdDict()
      self._BuildUrdfIds()
      if self._remove_default_joint_damping:
        self._RemoveDefaultJointDamping()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self._RecordInertiaInfoFromURDF()
      self.ResetPose(add_constraint=True)
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, init_position, INIT_ORIENTATION)
      self._pybullet_client.resetBaseVelocity(self.quadruped, INIT_VELOCITY, INIT_ANGULAR_VELOCITY)
      self.ResetPose(add_constraint=False)
    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors
    self._step_counter = 0

    # Perform reset motion within reset_duration if in position control mode.
    # Nothing is performed if in torque control mode for now.
    # TODO(jietan): Add reset motion when the torque control is fully supported.
    self._observation_history.clear()
    if not self._torque_control_enabled and reset_time > 0.0:
      self.ReceiveObservation()
      """
      for _ in range(100):
        self.ApplyAction([math.pi / 2] * self.num_motors)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
      """
      if default_motor_angles is not None:
        num_steps_to_reset = int(reset_time / self.time_step)
        for _ in range(num_steps_to_reset):
          self.ApplyAction(default_motor_angles, [0] * self.num_motors)
          self._pybullet_client.stepSimulation()
          self.ReceiveObservation()
    self.ReceiveObservation()

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.POSITION_CONTROL,
                                                targetPosition=desired_angle,
                                                positionGain=self._kp,
                                                velocityGain=self._kd,
                                                force=self._max_force)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name], desired_angle)

  def ResetPose(self, add_constraint):
    """Reset the pose of the minitaur.

    Args:
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    for i in range(self.num_legs):
      self._ResetPoseForLeg(i, add_constraint)

  def _ResetPoseForLeg(self, leg_id, add_constraint):
    """Reset the initial pose for the leg.

    Args:
      leg_id: It should be 0, 1, 2, or 3, which represents the leg at
        front_left, back_left, front_right and back_right.
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    knee_friction_force = 0
    half_pi = math.pi / 2.0
    knee_angle = -2.1834

    leg_position = LEG_POSITION[leg_id]
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["motor_" + leg_position + "L_joint"],
                                          self._motor_direction[2 * leg_id] * IP_Joint_Position[4 * leg_id],
                                          targetVelocity=0)
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["motor_" + leg_position + "R_joint"],
                                          self._motor_direction[2 * leg_id + 1] * IP_Joint_Position[4 * leg_id + 1],
                                          targetVelocity=0)
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["knee_" + leg_position + "L_link"],
                                          self._motor_direction[2 * leg_id] * -(math.pi - IP_Joint_Position[4 * leg_id + 2]),
                                          targetVelocity=0)
    self._pybullet_client.resetJointState(self.quadruped,
                                          self._joint_name_to_id["knee_" + leg_position + "R_link"],
                                          self._motor_direction[2 * leg_id + 1] * -(math.pi - IP_Joint_Position[4 * leg_id + 3]),
                                          targetVelocity=0)
    if add_constraint:
      self._pybullet_client.createConstraint(
          self.quadruped, self._joint_name_to_id["knee_" + leg_position + "R_link"],
          self.quadruped, self._joint_name_to_id["knee_" + leg_position + "L_link"],
          self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0], [0, 0.005, 0.2], [0, 0.01, 0.2])

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      # Disable the default motor in pybullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_" + leg_position + "L_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(self._joint_name_to_id["motor_" + leg_position + "R_joint"]),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=knee_friction_force)

    else:
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "L_joint",
                                       self._motor_direction[2 * leg_id] * half_pi)
      self._SetDesiredMotorAngleByName("motor_" + leg_position + "R_joint",
                                       self._motor_direction[2 * leg_id + 1] * half_pi)

    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

  def GetBasePosition(self):
    """Get the position of minitaur's base.

    Returns:
      The position of minitaur's base.
    """
    position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetTrueBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame.
    """
    orientation = self.GetTrueBaseOrientation()
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
    return np.asarray(roll_pitch_yaw)

  def GetBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
      and latency.
    """
    delayed_orientation = np.array(
        self._control_observation[3 * self.num_motors:3 * self.num_motors + 4])
    delayed_roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(delayed_orientation)
    roll_pitch_yaw = self._AddSensorNoise(np.array(delayed_roll_pitch_yaw),
                                          self._observation_noise_stdev[3])
    return roll_pitch_yaw

  def GetTrueMotorAngles(self):
    """Gets the eight motor angles at the current moment, mapped to [-pi, pi].

    Returns:
      Motor angles, mapped to [-pi, pi].
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorAngles(self):
    """Gets the eight motor angles.

    This function mimicks the noisy sensor reading and adds latency. The motor
    angles that are delayed, noise polluted, and mapped to [-pi, pi].

    Returns:
      Motor angles polluted by noise and latency, mapped to [-pi, pi].
    """
    motor_angles = self._AddSensorNoise(np.array(self._control_observation[0:self.num_motors]),
                                        self._observation_noise_stdev[0])
    return MapToMinusPiToPi(motor_angles)

  def GetTrueMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Velocities of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[self.num_motors:2 * self.num_motors]),
        self._observation_noise_stdev[1])

  def GetTrueMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques

  def GetMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Motor torques of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[2 * self.num_motors:3 * self.num_motors]),
        self._observation_noise_stdev[2])

  def GetTrueBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    Returns:
      The orientation of minitaur's base.
    """
    _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return orientation

  def GetBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      The orientation of minitaur's base polluted by noise and latency.
    """
    return self._pybullet_client.getQuaternionFromEuler(self.GetBaseRollPitchYaw())

  def GetTrueBaseVelocity(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base.
    """
    vel = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray([vel[0][0], vel[0][1], vel[0][2]])

  def GetTrueBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base.
    """
    vel = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray([vel[1][0], vel[1][1], vel[1][2]])

  def GetBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base polluted by noise
      and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[3 * self.num_motors + 4:3 * self.num_motors + 7]),
        self._observation_noise_stdev[4])

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def ApplyAction(self, angle_commands, velocity_commands, motor_kps=None, motor_kds=None):
    """Set the desired motor angles to the motors of the minitaur.

    The desired motor angles are clipped based on the maximum allowed velocity.
    If the pd_control_enabled is True, a torque is calculated according to
    the difference between current and desired joint angle, as well as the joint
    velocity. This torque is exerted to the motor. For more information about
    PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

    Args:
      motor_commands: The eight desired motor angles.
      motor_kps: Proportional gains for the motor model. If not provided, it
        uses the default kp of the minitaur for all the motors.
      motor_kds: Derivative gains for the motor model. If not provided, it
        uses the default kd of the minitaur for all the motors.
    """
    if self._motor_velocity_limit < np.inf:
      current_motor_angle = self.GetTrueMotorAngles()
      motor_commands_max = (current_motor_angle + self.time_step * self._motor_velocity_limit)
      motor_commands_min = (current_motor_angle - self.time_step * self._motor_velocity_limit)
      motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)
    # Set the kp and kd for all the motors if not provided as an argument.
    if motor_kps is None:
      motor_kps = np.full(8, self._kp)
    if motor_kds is None:
      motor_kds = np.full(8, self._kd)

    if self._accurate_motor_model_enabled or self._pd_control_enabled:
      true_motor_angle = self.GetTrueMotorAngles()
      true_motor_velocity = self.GetTrueMotorVelocities()
      if self._accurate_motor_model_enabled:
        actual_torque, observed_torque = self._motor_model.convert_to_torque(angle_commands, true_motor_angle, velocity_commands, true_motor_velocity, motor_kps, motor_kds)
        if self._motor_overheat_protection:
          for i in range(self.num_motors):
            if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
              self._overheat_counter[i] += 1
            else:
              self._overheat_counter[i] = 0
            if (self._overheat_counter[i] > OVERHEAT_SHUTDOWN_TIME / self.time_step):
              self._motor_enabled_list[i] = False

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = observed_torque

        # Transform into the motor space when applying the torque.
        self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)

        for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                         self._applied_motor_torque,
                                                         self._motor_enabled_list):
          if motor_enabled:
            self._SetMotorTorqueById(motor_id, motor_torque)
          else:
            self._SetMotorTorqueById(motor_id, 0)
      else:
        torque_commands = -1 * motor_kps * (q - motor_commands) - motor_kds * qdot

        # The torque is already in the observation space because we use
        # GetMotorAngles and GetMotorVelocities.
        self._observed_motor_torques = torque_commands

        # Transform into the motor space when applying the torque.
        self._applied_motor_torques = np.multiply(self._observed_motor_torques,
                                                  self._motor_direction)

        for motor_id, motor_torque in zip(self._motor_id_list, self._applied_motor_torques):
          self._SetMotorTorqueById(motor_id, motor_torque)
    else:
      motor_commands_with_direction = np.multiply(motor_commands, self._motor_direction)
      for motor_id, motor_command_with_direction in zip(self._motor_id_list,
                                                        motor_commands_with_direction):
        self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.

    Args:
      actions: The theta, phi of the leg model.
    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """
    motor_angle = copy.deepcopy(actions)
    scale_for_singularity = 1
    offset_for_singularity = 1.5
    half_num_motors = int(self.num_motors / 2)
    quater_pi = math.pi / 4
    for i in range(self.num_motors):
      action_idx = int(i // 2)
      forward_backward_component = (
          -scale_for_singularity * quater_pi *
          (actions[action_idx + half_num_motors] + offset_for_singularity))
      extension_component = (-1)**i * quater_pi * actions[action_idx]
      if i >= half_num_motors:
        extension_component = -extension_component
      motor_angle[i] = (math.pi + forward_backward_component + extension_component)
    return motor_angle

  def GetBaseMassesFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetBaseInertiasFromURDF(self):
    """Get the inertia of the base from the URDF file."""
    return self._base_inertia_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetLegInertiasFromURDF(self):
    """Get the inertia of the legs from the URDF file."""
    return self._leg_inertia_urdf

  def SetBaseMasses(self, base_mass):
    """Set the mass of minitaur's base.

    Args:
      base_mass: A list of masses of each body link in CHASIS_LINK_IDS. The
        length of this list should be the same as the length of CHASIS_LINK_IDS.
    Raises:
      ValueError: It is raised when the length of base_mass is not the same as
        the length of self._chassis_link_ids.
    """
    if len(base_mass) != len(self._chassis_link_ids):
      raise ValueError("The length of base_mass {} and self._chassis_link_ids {} are not "
                       "the same.".format(len(base_mass), len(self._chassis_link_ids)))
    for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
      self._pybullet_client.changeDynamics(self.quadruped, chassis_id, mass=chassis_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link masses, last 8 correspond
    to motor masses (24 total).

    Args:
      leg_masses: The leg and motor masses for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of masses is not equal to number
        of links + motors.
    """
    if len(leg_masses) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
      self._pybullet_client.changeDynamics(self.quadruped, leg_id, mass=leg_mass)
    motor_masses = leg_masses[len(self._leg_link_ids):]
    for link_id, motor_mass in zip(self._motor_link_ids, motor_masses):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=motor_mass)

  def SetBaseInertias(self, base_inertias):
    """Set the inertias of minitaur's base.

    Args:
      base_inertias: A list of inertias of each body link in CHASIS_LINK_IDS.
        The length of this list should be the same as the length of
        CHASIS_LINK_IDS.
    Raises:
      ValueError: It is raised when the length of base_inertias is not the same
        as the length of self._chassis_link_ids and base_inertias contains
        negative values.
    """
    if len(base_inertias) != len(self._chassis_link_ids):
      raise ValueError("The length of base_inertias {} and self._chassis_link_ids {} are "
                       "not the same.".format(len(base_inertias), len(self._chassis_link_ids)))
    for chassis_id, chassis_inertia in zip(self._chassis_link_ids, base_inertias):
      for inertia_value in chassis_inertia:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           chassis_id,
                                           localInertiaDiagonal=chassis_inertia)

  def SetLegInertias(self, leg_inertias):
    """Set the inertias of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link inertia, last 8 correspond
    to motor inertia (24 total).

    Args:
      leg_inertias: The leg and motor inertias for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of inertias is not equal to
      the number of links + motors or leg_inertias contains negative values.
    """

    if len(leg_inertias) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_inertia in zip(self._leg_link_ids, leg_inertias):
      for inertia_value in leg_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           leg_id,
                                           localInertiaDiagonal=leg_inertia)

    motor_inertias = leg_inertias[len(self._leg_link_ids):]
    for link_id, motor_inertia in zip(self._motor_link_ids, motor_inertias):
      for inertia_value in motor_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           localInertiaDiagonal=motor_inertia)

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, lateralFriction=foot_friction)

  # TODO(b/73748980): Add more API's to set other contact parameters.
  def SetFootRestitution(self, foot_restitution):
    """Set the coefficient of restitution at the feet.

    Args:
      foot_restitution: The coefficient of restitution (bounciness) of the feet.
        This value is shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(self.quadruped, link_id, restitution=foot_restitution)

  def SetJointFriction(self, joint_frictions):
    for knee_joint_id, friction in zip(self._foot_link_ids, joint_frictions):
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=knee_joint_id,
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=friction)

  def GetNumKneeJoints(self):
    return len(self._foot_link_ids)

  def SetBatteryVoltage(self, voltage):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    if self._accurate_motor_model_enabled:
      self._motor_model.set_viscous_damping(viscous_damping)

  def GetTrueObservation(self):
    observation = []
    observation.extend(self.GetTrueMotorAngles())
    observation.extend(self.GetTrueMotorVelocities())
    observation.extend(self.GetTrueMotorTorques())
    observation.extend(self.GetTrueBaseOrientation())
    observation.extend(self.GetTrueBaseRollPitchYawRate())
    return observation

  def ReceiveObservation(self):
    """Receive the observation from sensors.

    This function is called once per step. The observations are only updated
    when this function is called.
    """
    self._observation_history.appendleft(self.GetTrueObservation())
    self._control_observation = self._GetControlObservation()

  def _GetDelayedObservation(self, latency):
    """Get observation that is delayed by the amount specified in latency.

    Args:
      latency: The latency (in seconds) of the delayed observation.
    Returns:
      observation: The observation which was actually latency seconds ago.
    """
    if latency <= 0 or len(self._observation_history) == 1:
      observation = self._observation_history[0]
    else:
      n_steps_ago = int(latency / self.time_step)
      if n_steps_ago + 1 >= len(self._observation_history):
        return self._observation_history[-1]
      remaining_latency = latency - n_steps_ago * self.time_step
      blend_alpha = remaining_latency / self.time_step
      observation = ((1.0 - blend_alpha) * np.array(self._observation_history[n_steps_ago]) +
                     blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
    return observation

  def _GetPDObservation(self):
    pd_delayed_observation = self._GetDelayedObservation(self._pd_latency)
    q = pd_delayed_observation[0:self.num_motors]
    qdot = pd_delayed_observation[self.num_motors:2 * self.num_motors]
    return (np.array(q), np.array(qdot))

  def _GetControlObservation(self):
    control_delayed_observation = self._GetDelayedObservation(self._control_latency)
    return control_delayed_observation

  def _AddSensorNoise(self, sensor_values, noise_stdev):
    if noise_stdev <= 0:
      return sensor_values
    observation = sensor_values + np.random.normal(scale=noise_stdev, size=sensor_values.shape)
    return observation

  def SetControlLatency(self, latency):
    """Set the latency of the control loop.

    It measures the duration between sending an action from Nvidia TX2 and
    receiving the observation from microcontroller.

    Args:
      latency: The latency (in seconds) of the control loop.
    """
    self._control_latency = latency

  def GetControlLatency(self):
    """Get the control latency.

    Returns:
      The latency (in seconds) between when the motor command is sent and when
        the sensor measurements are reported back to the controller.
    """
    return self._control_latency

  def SetMotorGains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
    self._kp = kp
    self._kd = kd
    if self._accurate_motor_model_enabled:
      self._motor_model.set_motor_gains(kp, kd)

  def GetMotorGains(self):
    """Get the gains of the motor.

    Returns:
      The proportional gain.
      The derivative gain.
    """
    return self._kp, self._kd

  def SetMotorStrengthRatio(self, ratio):
    """Set the strength of all motors relative to the default value.

    Args:
      ratio: The relative strength. A scalar range from 0.0 to 1.0.
    """
    if self._accurate_motor_model_enabled:
      self._motor_model.set_strength_ratios([ratio] * self.num_motors)

  def SetMotorStrengthRatios(self, ratios):
    """Set the strength of each motor relative to the default value.

    Args:
      ratios: The relative strength. A numpy array ranging from 0.0 to 1.0.
    """
    if self._accurate_motor_model_enabled:
      self._motor_model.set_strength_ratios(ratios)

  def SetTimeSteps(self, action_repeat, simulation_step):
    """Set the time steps of the control and simulation.

    Args:
      action_repeat: The number of simulation steps that the same action is
        repeated.
      simulation_step: The simulation time step.
    """
    self.time_step = simulation_step
    self._action_repeat = action_repeat

  @property
  def chassis_link_ids(self):
    return self._chassis_link_ids

class MinitaurGymEnv(gym.Env):
  """The gym environment for the minitaur.

  It simulates the locomotion of a minitaur, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the minitaur walks in 1000 steps and penalizes the energy
  expenditure.

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}

  def __init__(self,
               urdf_root=pybullet_data.getDataPath(),
               urdf_version=None,
               distance_weight=1.0,
               energy_weight=0.005,
               shake_weight=0.0,
               drift_weight=0.0,
               distance_limit=float("inf"),
               observation_noise_stdev=SENSOR_NOISE_STDDEV,
               self_collision_enabled=True,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               leg_model_enabled=True,
               accurate_motor_model_enabled=False,
               remove_default_joint_damping=False,
               motor_kp=1.0,
               motor_kd=0.02,
               control_latency=0.0,
               pd_latency=0.0,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               hard_reset=True,
               on_rack=False,
               render=False,
               num_steps_to_log=1000,
               action_repeat=1,
               control_time_step=None,
               env_randomizer=None,
               forward_reward_cap=float("inf"),
               reflection=True,
               log_path=None):
    """Initialize the minitaur gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION,
        RAINBOW_DASH_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. DERPY_V0_URDF_VERSION
        is the result of first pass system identification for derpy.
        We will have a different URDF and related Minitaur class each time we
        perform system identification. While the majority of the code of the
        class remains the same, some code changes (e.g. the constraint location
        might change). __init__() will choose the right Minitaur class from
        different minitaur modules based on
        urdf_version.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      remove_default_joint_damping: Whether to remove the default joint damping.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      control_latency: It is the delay in the controller between when an
        observation is made at some point, and when that reading is reported
        back to the Neural Network.
      pd_latency: latency of the PD controller loop. PD calculates PWM based on
        the motor angle and velocity. The latency measures the time between when
        the motor angle and velocity are observed on the microcontroller and
        when the true state happens on the motor. It is typically (0.001-
        0.002s).
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the minitaur back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode that will
        be logged. If the number of steps is more than num_steps_to_log, the
        environment will still be running, but only first num_steps_to_log will
        be recorded in logging.
      action_repeat: The number of simulation steps before actions are applied.
      control_time_step: The time step between two successive control signals.
      env_randomizer: An instance (or a list) of EnvRandomizer(s). An
        EnvRandomizer may randomize the physical property of minitaur, change
          the terrrain during reset(), or add perturbation forces during step().
      forward_reward_cap: The maximum value that forward reward is capped at.
        Disabled (Inf) by default.
      log_path: The path to write out logs. For the details of logging, refer to
        minitaur_logging.proto.
    Raises:
      ValueError: If the urdf_version is not supported.
    """
    # Set up logging.
    self._log_path = log_path
    self.logging = ML.MinitaurLogging(log_path)
    # PD control needs smaller time step for stability.
    if control_time_step is not None:
      self.control_time_step = control_time_step
      self._action_repeat = action_repeat
      self._time_step = control_time_step / action_repeat
    else:
      # Default values for time step and action repeat
      if accurate_motor_model_enabled or pd_control_enabled:
        self._time_step = 0.002
        self._action_repeat = 5
      else:
        self._time_step = 0.01
        self._action_repeat = 1
      self.control_time_step = self._time_step * self._action_repeat
    # TODO(b/73829334): Fix the value of self._num_bullet_solver_iterations.
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
    self._episode_proto = ML2.MinitaurEpisode()
    if self._is_render:
      self._pybullet_client = BC.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = BC.BulletClient()
    if self._urdf_version is None:
      self._urdf_version = DEFAULT_URDF_VERSION
    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self.seed()
    self.reset()
    observation_high = (self._get_observation_upper_bound() + OBSERVATION_EPS)
    observation_low = (self._get_observation_lower_bound() - OBSERVATION_EPS)
    action_dim = NUM_MOTORS
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(observation_low, observation_high)
    self.viewer = None
    self._hard_reset = hard_reset  # This assignment need to be after reset()

  def close(self):
    if self._env_step_counter > 0:
      self.logging.save_episode(self._episode_proto)
    self.minitaur.Terminate()

  def add_env_randomizer(self, env_randomizer):
    self._env_randomizers.append(env_randomizer)

  def reset(self, initial_motor_angles=None, reset_duration=1.0):
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
    if self._env_step_counter > 0:
      self.logging.save_episode(self._episode_proto)
    self._episode_proto = ML2.MinitaurEpisode()
    ML.preallocate_episode_proto(self._episode_proto, self._num_steps_to_log)
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      if (self._reflection):
        self._pybullet_client.changeVisualShape(self._ground_id, -1, rgbaColor=[1, 1, 1, 0.8])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, self._ground_id)
      self._pybullet_client.setGravity(0, 0, -10)
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection
      self.minitaur = Minitaur(
            pybullet_client=self._pybullet_client,
            action_repeat=self._action_repeat,
            urdf_root=self._urdf_root,
            time_step=self._time_step,
            self_collision_enabled=self._self_collision_enabled,
            motor_velocity_limit=self._motor_velocity_limit,
            pd_control_enabled=self._pd_control_enabled,
            accurate_motor_model_enabled=acc_motor,
            remove_default_joint_damping=self._remove_default_joint_damping,
            motor_kp=self._motor_kp,
            motor_kd=self._motor_kd,
            control_latency=self._control_latency,
            pd_latency=self._pd_latency,
            observation_noise_stdev=self._observation_noise_stdev,
            torque_control_enabled=self._torque_control_enabled,
            motor_overheat_protection=motor_protect,
            on_rack=self._on_rack)
    self.minitaur.Reset(reload_urdf=False,
                        default_motor_angles=initial_motor_angles,
                        reset_time=reset_duration)

    # Loop over all env randomizers.
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_env(self)

    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                     self._cam_pitch, [0, 0, 0])
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
    return self._get_observation()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError("{}th action {} out of bounds.".format(i, action_component))
      action = self.minitaur.ConvertFromLegModel(action)
    return action

  def step(self, action, action_dot):
    """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    self._last_base_position = self.minitaur.GetBasePosition()

    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self.control_time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
      base_pos = self.minitaur.GetBasePosition()
      # Keep the previous orientation of the camera set by the user.
      [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
      self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_step(self)

    action = self._transform_action_to_motor_command(action)
    self.minitaur.Step(action, action_dot)
    reward = self._reward()
    done = self._termination()
    if self._log_path is not None:
      ML.update_episode_proto(self._episode_proto, self.minitaur, action,
                                            self._env_step_counter)
    self._env_step_counter += 1
    if done:
      self.minitaur.Terminate()
    return np.array(self._get_true_observation())

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.minitaur.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def get_minitaur_motor_angles(self):
    """Get the minitaur's motor angles.

    Returns:
      A numpy array of motor angles.
    """
    return np.array(self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:MOTOR_ANGLE_OBSERVATION_INDEX +  NUM_MOTORS])

  def get_minitaur_motor_velocities(self):
    """Get the minitaur's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
    return np.array(
        self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS])

  def get_minitaur_motor_torques(self):
    """Get the minitaur's motor torques.

    Returns:
      A numpy array of motor torques.
    """
    return np.array(
        self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS])

  def get_minitaur_base_orientation(self):
    """Get the minitaur's base orientation, represented by a quaternion.

    Returns:
      A numpy array of minitaur's orientation.
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  def is_fallen(self):
    """Decide whether the minitaur has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the minitaur is considered fallen.

    Returns:
      Boolean value that indicates whether the minitaur has fallen.
    """
    orientation = self.minitaur.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self.minitaur.GetBasePosition()
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < 0.13)

  def _termination(self):
    position = self.minitaur.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self.is_fallen() or distance > self._distance_limit

  def _reward(self):
    current_base_position = self.minitaur.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    # Cap the forward reward if a cap is set.
    forward_reward = min(forward_reward, self._forward_reward_cap)
    # Penalty for sideways translation.
    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    # Penalty for sideways rotation of the body.
    orientation = self.minitaur.GetBaseOrientation()
    rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
    local_up_vec = rot_matrix[6:]
    shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
    energy_reward = -np.abs(
        np.dot(self.minitaur.GetMotorTorques(),
               self.minitaur.GetMotorVelocities())) * self._time_step
    objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
    weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
    reward = sum(weighted_objectives)
    self._objectives.append(objectives)
    return reward

  def get_objectives(self):
    return self._objectives

  @property
  def objective_weights(self):
    """Accessor for the weights for all the objectives.

    Returns:
      List of floating points that corresponds to weights for the objectives in
      the order that objectives are stored.
    """
    return self._objective_weights

  def _get_observation(self):
    """Get observation of this environment, including noise and latency.

    The minitaur class maintains a history of true observations. Based on the
    latency, this function will find the observation at the right time,
    interpolate if necessary. Then Gaussian noise is added to this observation
    based on self.observation_noise_stdev.

    Returns:
      The noisy observation with latency.
    """

    observation = []
    observation.extend(self.minitaur.GetMotorAngles().tolist())
    observation.extend(self.minitaur.GetMotorVelocities().tolist())
    observation.extend(self.minitaur.GetMotorTorques().tolist())
    observation.extend(list(self.minitaur.GetBaseOrientation()))
    self._observation = observation
    return self._observation

  def _get_true_observation(self):
    """Get the observations of this environment.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.minitaur.GetTrueMotorAngles().tolist())
    observation.extend(self.minitaur.GetTrueMotorVelocities().tolist())
    observation.extend(self.minitaur.GetTrueMotorTorques().tolist())
    observation.extend(list(self.minitaur.GetTrueBaseOrientation()))

    self._true_observation = observation
    return self._true_observation

  def _get_observation_upper_bound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.zeros(self._get_observation_dimension())
    num_motors = self.minitaur.num_motors
    upper_bound[0:num_motors] = math.pi  # Joint angle.
    upper_bound[num_motors:2 * num_motors] = MOTOR_SPEED_LIMIT  # Joint velocity.
    upper_bound[2 * num_motors:3 * num_motors] = OBSERVED_TORQUE_LIMIT  # Joint torque.
    upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
    return upper_bound

  def _get_observation_lower_bound(self):
    """Get the lower bound of the observation."""
    return -self._get_observation_upper_bound()

  def _get_observation_dimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self._get_observation())

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

  def set_time_step(self, control_step, simulation_step=0.001):
    """Sets the time step of the environment.

    Args:
      control_step: The time period (in seconds) between two adjacent control
        actions are applied.
      simulation_step: The simulation time step in PyBullet. By default, the
        simulation step is 0.001s, which is a good trade-off between simulation
        speed and accuracy.
    Raises:
      ValueError: If the control step is smaller than the simulation step.
    """
    if control_step < simulation_step:
      raise ValueError("Control step should be larger than or equal to simulation step.")
    self.control_time_step = control_step
    self._time_step = simulation_step
    self._action_repeat = int(round(control_step / simulation_step))
    self._num_bullet_solver_iterations = (NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
    self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=self._num_bullet_solver_iterations)
    self._pybullet_client.setTimeStep(self._time_step)
    self.minitaur.SetTimeSteps(action_repeat=self._action_repeat, simulation_step=self._time_step)

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def ground_id(self):
    return self._ground_id

  @ground_id.setter
  def ground_id(self, new_ground_id):
    self._ground_id = new_ground_id

  @property
  def env_step_counter(self):
    return self._env_step_counter

def main(argv):
    del argv
    try:
        env = MinitaurGymEnv(
            urdf_version="default",
            control_time_step=0.006,
            action_repeat=6,
            pd_latency=0.003,
            control_latency=FLAGS.control_latency,
            motor_kp=FLAGS.motor_kp,
            motor_kd=FLAGS.motor_kd,
            remove_default_joint_damping=True,
            leg_model_enabled=False,
            render=True,
            on_rack=False,
            accurate_motor_model_enabled=True,
            log_path=FLAGS.log_path)
        env.reset()

        controller = minitaur_bounding_controller.MinitaurRaibertBoundingController(env.minitaur)

        num_iter = range(1000)
        tstart = env.minitaur.GetTimeSinceReset()
        for i in num_iter:
            print('iteration number = ', i)
            t = env.minitaur.GetTimeSinceReset() - tstart
            phase, event = controller.update(t)

            action, action_dot = controller.get_action(Inputs_Joint_Position_Front_Stance, Inputs_Joint_Position_Back_Stance, Inputs_Joint_Velocity_Front_Stance, Inputs_Joint_Velocity_Back_Stance)
            q_true = env.step(action, action_dot)
            #print('front actual swing = ', (q_true[1]-q_true[0])/2)
            #print('front actual exten = ', (q_true[1]+q_true[0])/2)
            #print('back actual swing = ', (q_true[3]-q_true[2])/2)
            #print('back actual exten = ',(q_true[3]+q_true[2])/2)
            front_left_leg_swing_actual.append((q_true[1]-q_true[0])/2)
            front_left_leg_exten_actual.append((q_true[1]+q_true[0])/2)
            front_left_leg_swing_desire.append((action[1]-action[0])/2)
            front_left_leg_exten_desire.append((action[1]+action[0])/2)
            input('-------------Pause-------------')

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(num_iter, front_left_leg_swing_actual, num_iter, front_left_leg_swing_desire)
        plt.ylabel('Swing')
        plt.subplot(2,1,2)
        plt.plot(num_iter, front_left_leg_exten_actual, num_iter, front_left_leg_exten_desire)
        plt.ylabel('Extension')
        plt.draw()
    finally:
        env.close()
        plt.show()

if __name__ == "__main__":
    tf.compat.v1.app.run(main)
