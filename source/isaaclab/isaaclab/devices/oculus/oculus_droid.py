"Joystick controller using OculusReader input."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni
import time
import math
import ipdb
from ..device_base import DeviceBase

# from .oculus_reader import OculusReader
from .FPS_counter import FPSCounter
from .buttons_parser import parse_buttons
import threading
import os
from ppadb.client import Client as AdbClient
import sys
import multiprocessing
import subprocess
import threading

def eprint(*args, **kwargs):
    RED = "\033[1;31m"  
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)

def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

def quat_to_euler(quat, degrees=False):
    euler = Rotation.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler

def euler_to_quat(euler, degrees=False):
    return Rotation.from_euler("xyz", euler, degrees=degrees).as_quat()

def rmat_to_euler(rot_mat, degrees=False):
    euler = Rotation.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler

def euler_to_rmat(euler, degrees=False):
    return Rotation.from_euler("xyz", euler, degrees=degrees).as_matrix()

def rmat_to_quat(rot_mat, degrees=False):
    quat = Rotation.from_matrix(rot_mat).as_quat()
    return quat

def quat_diff(target, source):
    result = Rotation.from_quat(target) * Rotation.from_quat(source).inv()
    return result.as_quat()

def add_angles(delta, source, degrees=False):
    delta_rot = Rotation.from_euler("xyz", delta, degrees=degrees)
    source_rot = Rotation.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


class OculusReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.rail.oculus.teleop',
            print_FPS=False,
            run=True
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        self._lock = threading.Lock()
        self.tag = 'wE9ryARX'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell('am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER')
        self.thread = threading.Thread(target=self.device.shell, args=("logcat -T 0", self.read_logcat_by_line))
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                os.system('adb tcpip ' + str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            os.system('adb devices')
            devices = client.devices()
        for device in devices:
            if device.serial.count('.') < 3:
                return device
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        exit(1)

    def get_device(self):
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'teleop-debug.apk')
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split('&')
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split('|')
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4,4))
            pair = pair_string.split(':')
            if len(pair) != 2:
                continue
            left_right_char = pair[0] # is r or l
            transform_string = pair[1]
            values = transform_string.split(' ')
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons
    
    def get_valid_transforms_and_buttons(self):
        while True:
            transforms, buttons = self.get_transformations_and_buttons()
        
            # Check if 'l' and 'r' are in transforms, indicating valid data
            if "l" in transforms and "r" in transforms:
                # print("Valid transforms received.")
                return transforms, buttons
        
            # Optionally log or print when data isn't available
            print("Waiting for valid transforms...")
        
            # Wait a bit before trying again (to avoid busy loop)
            time.sleep(0.1)  # Sleep for 100ms before retrying
    
    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()

class Oculus_mobile(DeviceBase):
    """A joystick controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a joystick controller for a bi-manual mobile manipulator.
    It uses the rail oculus reader to listen to joystick events and map them to robot's
    task-space commands.

    The command comprises of three parts:

    * delta control for mobile base: a 3D vector of (vx, vy, vz) in meters per second.
    * delta pose for each arms: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper for each arms: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= 
        Description                    Joystick event    
        ============================== ================= 
        Toggle gripperR (open/close)   RTr (index finger)
        Toggle gripperL (open/close)   LTr (index finger)
        Move along xy plane            leftJS xy position
        Rotate along yaw               RightJS x         
        Move right arm                 IK of right joystick SE(3) 4x4 matrix
        Move left arm                  IK of left joystick SE(3) 4x4 matrix
        ============================== ================= 
    """
    def __init__(self, pos_sensitivity: float = 0.01, rot_sensitivity: float = 0.01, base_sensitivity: float = 0.05):
        """
        Args:
            pos_sensitivity: Magnitude of input position command scaling for arms.
            rot_sensitivity: Magnitude of input Rotation command scaling for arms.
            base_sensitivity: Magnitude of input command scaling for the mobile base.
        """
        # sensitivities
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_sensitivity = base_sensitivity

        # initialize OculusReader for joystick input
        self.oculus_reader = OculusReader()

        # set up yaw cumulative motion
        self._key_hold_start = {}  # Track when rightJS is moved
        self._base_z_accum = 0.0   # Accumulated vertical motion for base

        # command buffers
        self._close_gripper_left = False
        self._close_gripper_right = False
        self._delta_pos_left = np.zeros(3)  # (x, y, z) for left arm
        self._delta_rot_left = np.zeros(3)  # (roll, pitch, yaw) for left arm
        self._delta_pos_right = np.zeros(3)  # (x, y, z) for right arm
        self._delta_rot_right = np.zeros(3)  # (roll, pitch, yaw) for right arm
        self._delta_base = np.zeros(3)  # (x, y, yaw) for mobile base
        
        # arm
        self._last_transform_left = np.eye(4)
        self._last_transform_right = np.eye(4)

        # gripper
        self._prev_LTr_state = False
        self._prev_RTr_state = False

        # xy
        self._last_leftJS = (0.0, 0.0)
        self._js_threshold = 0.4  # tune this to taste
        
        # yaw
        self.Rotation_divisor = 1.2037685675
        self.base_rot_sensitivity = 10
        
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self.oculus_reader.stop()
    
    def reset(self):
        """Reset all commands."""
        # self._close_gripper_left = False
        # self._close_gripper_right = False
        self._delta_pos_left = np.zeros(3)
        self._delta_rot_left = np.zeros(3)
        self._delta_pos_right = np.zeros(3)
        self._delta_rot_right = np.zeros(3)
        self._delta_base = np.zeros(3)
        # self._base_z_accum = 0.0
        # self._key_hold_start = {}  # Reset key hold tracking

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Joystick Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"


    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func


    def advance(self) -> tuple[np.ndarray, bool, np.ndarray, bool, np.ndarray]:
        """
        Read joystick events and return command for BMM.

        Threshold:
            base xy: 0.1
            base z: 0.7

        Returns:
            A tuple containing the delta pose commands for left arm, right arm, and mobile base.
        """
        # fetch latest controller data
        transforms, buttons = self.oculus_reader.get_valid_transforms_and_buttons()  # returns dict with 'leftJS', 'rightJS'

        # Delta pose for arms

        # 1. grab current
        T_l = transforms["l"]
        T_r = transforms["r"]

        # 2. TRANSLATION DELTA: just position difference
        #    (new_pos - old_pos), then axis‐swap & sensitivity
        dp_l_raw = T_l[:3, 3] - self._last_transform_left[:3, 3]
        dp_r_raw = T_r[:3, 3] - self._last_transform_right[:3, 3]

        self._delta_pos_left  = np.array([-dp_l_raw[2], -dp_l_raw[0], dp_l_raw[1]]) * self.pos_sensitivity
        self._delta_pos_right = np.array([-dp_r_raw[2], -dp_r_raw[0], dp_r_raw[1]]) * self.pos_sensitivity

        # 3. Rotation DELTA: same as before, via delta‐matrix
        R_l_delta = T_l[:3, :3] @ self._last_transform_left[:3, :3].T
        R_r_delta = T_r[:3, :3] @ self._last_transform_right[:3, :3].T

        rv_l = Rotation.from_matrix(R_l_delta).as_rotvec() * self.rot_sensitivity
        rv_r = Rotation.from_matrix(R_r_delta).as_rotvec() * self.rot_sensitivity

        # re‐order [z, x, y] and flip signs on first two
        self._delta_rot_left  = rv_l[[2, 0, 1]] * np.array([-1, -1, 1])
        self._delta_rot_right = rv_r[[2, 0, 1]] * np.array([-1, -1, 1])

        # 4. save current for next frame
        self._last_transform_left  = T_l.copy()
        self._last_transform_right = T_r.copy()

        # 5. reset on button X
        if buttons.get("X", False):
            self.reset()

        # Then in your update loop or callback:
        if buttons['LTr'] and not self._prev_LTr_state:
            self._close_gripper_left = not self._close_gripper_left
        
        # Update the previous state for the next cycle
        self._prev_LTr_state = buttons['LTr']

        if buttons['RTr'] and not self._prev_RTr_state:
            self._close_gripper_right = not self._close_gripper_right
        # Update the previous state for the next cycle
        self._prev_RTr_state = buttons['RTr']

        # mobile base

        # yaw
        # check if the rightJS is moved to right
        if buttons['rightJS'][0] < -0.7  and ('counterclockwise' not in self._key_hold_start):
            self._delta_base[2] += self.base_sensitivity * self.base_rot_sensitivity
            self._key_hold_start['counterclockwise'] = time.time()

        # check if the rightJS is moved to left
        elif buttons['rightJS'][0] > 0.7 and ('clockwise' not in self._key_hold_start):
            self._delta_base[2] -= self.base_sensitivity * self.base_rot_sensitivity
            self._key_hold_start['clockwise'] = time.time()

        # check if the rightJS is returned to the center (similar to key realeased)
        elif buttons['rightJS'][0] == 0.0 and (('counterclockwise' in self._key_hold_start) or ('clockwise' in self._key_hold_start)):
            # remove the key from the dictionary
            if 'counterclockwise' in self._key_hold_start:
                duration = time.time() - self._key_hold_start['counterclockwise']
                self._base_z_accum += self.base_sensitivity * duration
                self._delta_base[2] = 0.0
                del self._key_hold_start['counterclockwise']
                
                
            if 'clockwise' in self._key_hold_start:
                duration = time.time() - self._key_hold_start['clockwise']
                self._base_z_accum -= self.base_sensitivity * duration
                self._delta_base[2] = 0.0
                del self._key_hold_start['clockwise']
        
        # xy
        if buttons['rightJS'][0] == 0.0:
            raw_x, raw_y = buttons['leftJS']
            new_js = (raw_x, raw_y)

            # compute Euclidean change
            dx = raw_x - self._last_leftJS[0]
            dy = raw_y - self._last_leftJS[1]

            # check if the leftJS is moved by the self._js_threshold
            if (dx*dx + dy*dy)**0.6 > self._js_threshold:
                theta = self._base_z_accum * self.base_rot_sensitivity / self.Rotation_divisor
                print(theta)
                # 1) subtract out the old
                ox, oy = self._last_leftJS
                old_vec = (
                    ox * np.asarray([
                        math.cos((theta)),
                        math.sin((theta)),
                        0.0
                    ]) +
                    oy * np.asarray([
                        -math.sin((theta)),
                        math.cos((theta)),
                        0.0
                    ])
                ) * self.base_sensitivity
                self._delta_base -= old_vec[[1,0,2]]* np.array([1, -1, 1])

                # 2) add in the new
                new_vec = (
                    raw_x * np.asarray([
                        math.cos((theta)),
                        math.sin((theta)),
                        0.0
                    ]) +
                    raw_y * np.asarray([
                        -math.sin((theta)),
                        math.cos((theta)),
                        0.0
                    ])
                ) * self.base_sensitivity
                self._delta_base += new_vec[[1,0,2]]* np.array([1, -1, 1])

                # 3) remember it
                self._last_leftJS = new_js
            
        # return the commands
        return (
            np.concatenate([self._delta_pos_left, self._delta_rot_left]),  # Left arm
            self._close_gripper_left,  # Left gripper
            np.concatenate([self._delta_pos_right, self._delta_rot_right ]),  # Right arm
            self._close_gripper_right,  # Right gripper
            self._delta_base,  # Mobile base
        )
    #  ({'l': array([[-0.224735 ,  0.415998 , -0.881158 , -0.0416255],
    #    [ 0.937851 , -0.153066 , -0.311457 , -0.103081 ],
    #    [-0.264441 , -0.89639  , -0.355745 ,  0.0854378],
    #    [ 0.       ,  0.       ,  0.       ,  1.       ]]), 
       
    #    'r': array([[-0.753894 ,  0.601267 ,  0.264804 , -0.0686894],
    #    [-0.130283 ,  0.258232 , -0.957258 , -0.0819744],
    #    [-0.643948 , -0.756171 , -0.116345 ,  0.0297576],
    #    [ 0.       ,  0.       ,  0.       ,  1.       ]])}, {'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': False, 'RTr': False, 'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)})

class Oculus_droid(DeviceBase):
    def __init__(
        self,
        max_lin_vel=1,
        max_rot_vel=1,
        max_gripper_vel=1,
        spatial_coeff=1,
        pos_action_gain=5,
        rot_action_gain=2,
        gripper_action_gain=3,
        rmat_reorder=[-2, -1, -3, 4],
        pos_sensitivity: float = 0.01, 
        rot_sensitivity: float = 0.01, 
        base_sensitivity: float = 0.05
    ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = {"r": np.eye(4), "l": np.eye(4)}
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)

        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain

        self.reset_orientation = {"r": True, "l": True}
        self.reset_state()

        run_threaded_command(self._update_internal_state)
    def get_valid_transforms_and_buttons(self):
        while True:
            transforms, buttons = self.get_transformations_and_buttons()
        
            # Check if 'l' and 'r' are in transforms, indicating valid data
            if "l" in transforms and "r" in transforms:
                # print("Valid transforms received.")
                return transforms, buttons
        
            # Optionally log or print when data isn't available
            print("Waiting for valid transforms...")
        
            # Wait a bit before trying again (to avoid busy loop)
            time.sleep(0.1)  # Sleep for 100ms before retrying
    def reset(self):
        # Stub for now
        print("Oculus_droid reset called")
        self.reset_orientation = {"r": True, "l": True}
        self.reset_state()

    def add_callback(self, callback):
        # Stub for now
        print("Callback registered but not used in Oculus_droid")

    def reset_state(self):
        self._state = {
            "poses": {},
            "buttons": {},
            "movement_enabled": {"r": False, "l": False},
            "controller_on": {"r": True, "l": True},
        }
        self.update_sensor = {"r": True, "l": True}
        self.reset_origin = {"r": True, "l": True}
        self.robot_origin = {"r": None, "l": None}
        self.vr_origin = {"r": None, "l": None}
        self.vr_state = {"r": None, "l": None}

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            time.sleep(1 / hz)
            poses, buttons = self.oculus_reader.get_valid_transforms_and_buttons()
            time_since_read = time.time() - last_read_time
            
            for cid in ["r", "l"]:
                self._state["controller_on"][cid] = time_since_read < num_wait_sec

                if cid + "G" in buttons:
                    toggled = self._state["movement_enabled"][cid] != buttons[cid + "G"]
                    self.update_sensor[cid] = self.update_sensor[cid] or buttons[cid + "G"]
                    self.reset_orientation[cid] = self.reset_orientation[cid] or buttons[cid + "J"]
                    self.reset_origin[cid] = self.reset_origin[cid] or toggled
                    self._state["movement_enabled"][cid] = buttons[cid + "G"]

                if cid in poses and self.reset_orientation[cid]:
                    rot_mat = np.asarray(poses[cid])
                    try:
                        rot_mat = np.linalg.inv(rot_mat)
                    except:
                        rot_mat = np.eye(4)
                        self.reset_orientation[cid] = True
                    self.vr_to_global_mat[cid] = rot_mat
                    if buttons.get(cid + "J") or self._state["movement_enabled"][cid]:
                        self.reset_orientation[cid] = False

            self._state["poses"] = poses
            self._state["buttons"] = buttons
            last_read_time = time.time()

    def _process_reading(self, cid):
        rot_mat = np.asarray(self._state["poses"][cid])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat[cid] @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        trig_key = "rightTrig" if cid == "r" else "leftTrig"
        vr_gripper = self._state["buttons"].get(trig_key, [0])[0]
        self.vr_state[cid] = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        lin_vel = lin_vel * min(1, self.max_lin_vel / (np.linalg.norm(lin_vel) + 1e-6))
        rot_vel = rot_vel * min(1, self.max_rot_vel / (np.linalg.norm(rot_vel) + 1e-6))
        gripper_vel = np.clip(gripper_vel, -self.max_gripper_vel, self.max_gripper_vel)
        return lin_vel, rot_vel, gripper_vel

    def _calculate_arm_action(self, cid, state_dict):
        if self.update_sensor[cid]:
            self._process_reading(cid)
            self.update_sensor[cid] = False
        state_dict = state_dict[0]  # This gives you a 1D tensor with shape (7,)

        robot_pos = state_dict[:3].cpu().numpy()  # Position (x, y, z)
        robot_euler = state_dict[3:6].cpu().numpy()  # Euler angles (roll, pitch, yaw)
        robot_quat = euler_to_quat(robot_euler)  # Quaternion from Euler angles
        robot_gripper = state_dict[6].item()  # Gripper state (scalar)


        if self.reset_origin[cid]:
            self.robot_origin[cid] = {"pos": robot_pos, "quat": robot_quat}
            self.vr_origin[cid] = {"pos": self.vr_state[cid]["pos"], "quat": self.vr_state[cid]["quat"]}
            self.reset_origin[cid] = False

        pos_action = (self.vr_state[cid]["pos"] - self.vr_origin[cid]["pos"]) - (
            robot_pos - self.robot_origin[cid]["pos"]
        )

        quat_action = quat_diff(
            quat_diff(self.vr_state[cid]["quat"], self.vr_origin[cid]["quat"]),
            quat_diff(robot_quat, self.robot_origin[cid]["quat"]),
        )
        euler_action = quat_to_euler(quat_action)

        gripper_action = (self.vr_state[cid]["gripper"] * 1.5) - robot_gripper

        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        gripper_action *= self.gripper_action_gain

        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)
        return np.concatenate([lin_vel, rot_vel, [gripper_vel]])

    def advance(self, obs_dict):
        if not self._state["movement_enabled"]:
            return (
            np.zeros(6),  # pose_L
            0.0,          # gripper_command_L
            np.zeros(6),  # pose_R
            0.0,          # gripper_command_R
            np.zeros(3),  # delta_pose_base
        )
        print(self._state)
        action_r = self._calculate_arm_action("r", obs_dict["right_arm"])
        action_l = self._calculate_arm_action("l", obs_dict["left_arm"])
        
        return (
        action_l[:6],  # pose_L
        action_l[6],   # gripper_command_L
        action_r[:6],  # pose_R
        action_r[6],   # gripper_command_R
        np.zeros(3),   # delta_pose_base
        )    

    def get_info(self):
        return {
            "success": self._state["buttons"].get("A", False),
            "failure": self._state["buttons"].get("B", False),
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }
