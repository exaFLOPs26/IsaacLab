"Joystick controller using OculusReader input."""
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/teleoperation/teleop_anubis.py --enable_cameras
import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni
import time
import math

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
import ipdb
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
        i=0
        while True:
            transforms, buttons = self.get_transformations_and_buttons()
        
            # Check if 'l' and 'r' are in transforms, indicating valid data
            if "l" in transforms and "r" in transforms:
                # print("Valid transforms received.")
                return transforms, buttons
        
            # Optionally log or print when data isn't available
            print("Waiting for valid transforms...")
            i+=1
            if i == 300:
                eprint("No valid transforms received after 20 seconds. Please check the connection.")
                sys.exit(1)
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
        pos_sensitivity: float = 0.03, 
        rot_sensitivity: float = 0.01, 
        base_sensitivity: float = 0.05
    ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = {"R": np.eye(4), "L": np.eye(4)}
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)

        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_sensitivity = base_sensitivity

        self.reset_orientation = {"R": True, "L": True}
        self.reset()
        
        self._additional_callbacks = dict()

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
        print("Oculus_droid reset called")
        self._state = {
            "poses": {},
            "buttons": {},
            "movement_enabled": {"R": False, "L": False},
            "controller_on": {"R": True, "L": True},
        }
        self.update_sensor = {"R": True, "L": True}
        self.reset_origin = {"R": True, "L": True}
        self.robot_origin = {"R": None, "L": None}
        self.vr_origin = {"R": None, "L": None}
        self.vr_state = {"R": None, "L": None}

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            time.sleep(1 / hz)
            
            # Read controler
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_valid_transforms_and_buttons()
            
            if poses == {}:
                continue
            
            if buttons["B"]:
                self._additional_callbacks['B']()
            
            if buttons["A"]:
                time.sleep(1)
                while not buttons["A"]:
                    time.sleep(0.5)
                time.sleep(1)
                self._additional_callbacks['A']()
            
            for cid in ["R", "L"]:
                self._state["controller_on"][cid] = time_since_read < num_wait_sec

                toggled = self._state["movement_enabled"][cid] != buttons[cid + "G"]
                self.update_sensor[cid] = self.update_sensor[cid] or buttons[cid + "G"]
                self.reset_orientation[cid] = self.reset_orientation[cid] or buttons.get(cid + "J", False)
                self.reset_origin[cid] = self.reset_origin[cid] or toggled
                
                self._state["movement_enabled"][cid] = buttons[cid + "G"]

                stop_updating = buttons[cid + "J"] or self._state["movement_enabled"][cid]

                if self.reset_orientation[cid]:
                    rot_mat = np.asarray(poses[cid.lower()])
                    if stop_updating:
                        self.reset_orientation[cid] = False
                    try:
                        rot_mat = np.linalg.inv(rot_mat)
                    except:
                        rot_mat = np.eye(4)
                        self.reset_orientation[cid] = True
                    self.vr_to_global_mat[cid] = rot_mat

            self._state["poses"] = poses
            self._state["buttons"] = buttons
            last_read_time = time.time()
    

    def _process_reading(self, cid):
        rot_mat = np.asarray(self._state["poses"][cid.lower()])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat[cid] @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        trig_key = "rightTrig" if cid == "R" else "leftTrig"
        vr_gripper = self._state["buttons"].get(trig_key, [0])[0]
        self.vr_state[cid] = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        lin_vel = lin_vel * min(1, self.max_lin_vel / (np.linalg.norm(lin_vel) + 1e-6))
        rot_vel = rot_vel * min(1, self.max_rot_vel / (np.linalg.norm(rot_vel) + 1e-6))

        gripper_vel = np.clip(gripper_vel, -self.max_gripper_vel, self.max_gripper_vel)

        if gripper_vel > -0.85:
            gripper_vel = 1
        elif gripper_vel < -0.85:
            gripper_vel = -1
        
        return lin_vel, rot_vel, gripper_vel

    def _calculate_arm_action(self, cid, state_dict):
        if self.update_sensor[cid]:
            self._process_reading(cid)
            self.update_sensor[cid] = False
        state_dict = state_dict[0]  # This gives you a 1D tensor with shape (7,)

        robot_pos = state_dict[:3].cpu().numpy()  # Position (x, y, z)
        # robot_euler = state_dict[3:6].cpu().numpy()  # Euler angles (roll, pitch, yaw)
        # robot_quat = euler_to_quat(robot_euler)  # Quaternion from Euler angles
        robot_quat = state_dict[3:7].cpu().numpy()
        robot_gripper = state_dict[7].item()  # Gripper state (scalar)

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
        euler_action = Rotation.from_quat(quat_action).as_rotvec() 
        

        # print("vr Right pos difference",(self.vr_state['R']["pos"] - self.vr_origin['R']["pos"]))
        # print("vr Right euler difference", quat_to_euler(quat_diff(self.vr_state['R']["quat"], self.vr_origin['R']["quat"])))

        gripper_action = ((1 - self.vr_state[cid]["gripper"]) * 1.7 ) + robot_gripper -0.036 # it was - robot_gripper before
        # if cid == "L":
        #     print("robot_left_gripper", robot_gripper)
        #     print("vr_left_gripper", self.vr_state[cid]["gripper"])
        #     print("left_gripper_action", gripper_action)
        # if cid == "R":
        #     print("robot_right_gripper", robot_gripper)
        #     print("vr_right_gripper", self.vr_state[cid]["gripper"])
        #     print("right_gripper_action", gripper_action)
        # pos_action *= self.pos_action_gain
        # euler_action *= self.rot_action_gain
        # gripper_action *= self.gripper_action_gain
        # print("------------------------")
        # print("pos_action", pos_action)
        # print("euler_action", euler_action)
        # print("gripper_action", gripper_action)
        # print("------------------------")

        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)
        # ipdb.set_trace()
        return np.concatenate([lin_vel, rot_vel, [gripper_vel]])
    
    def add_callback(self, button_name: str, func: Callable):
        """Register fn() to be called whenever button_name is pressed."""
        self._additional_callbacks[button_name] = func
        print(f"Callback for {button_name} registered.")
    
    def advance(self, obs_dict):

        if self._state["poses"] == {}:
            return (
            np.zeros(6),  # pose_L
            0.0,          # gripper_command_L
            np.zeros(6),  # pose_R
            0.0,          # gripper_command_R
            np.zeros(3),  # delta_pose_base
        )
        if self._state["movement_enabled"]["L"]:
            action_l = self._calculate_arm_action("L", obs_dict["left_arm"])
            action_l[:3] *= self.pos_sensitivity   
            action_l[3:6] *= self.rot_sensitivity 
            action_l[6] *= 1.0       
        else:
            action_l = np.zeros(7)

        if self._state["movement_enabled"]["R"]:
            action_r = self._calculate_arm_action("R", obs_dict["right_arm"])
            action_r[:3] *= self.pos_sensitivity
            action_r[3:6] *= self.rot_sensitivity
            action_r[6] *= 1.0
        else:
            action_r = np.zeros(7)
        
        return (
        action_l[:6],  # pose_L
        action_l[6],   # gripper_command_L
        action_r[:6],  # pose_R
        action_r[6],   # gripper_command_R
        np.zeros(3),   # delta_pose_base
        )    

    def advance_onearm(self, obs_dict):

        if self._state["poses"] == {}:
            return (
            np.zeros(6),  # pose_R
            0.0,          # gripper_command_R
        )

        if self._state["movement_enabled"]["R"]:
            action_r = self._calculate_arm_action("R", obs_dict["right_arm"])
            action_r[:3] *= self.pos_sensitivity
            action_r[3:6] *= self.rot_sensitivity
            action_r[6] *= 1.0
        else:
            action_r = np.zeros(7)
        
        return (
        action_r[:6],  # pose_R
        action_r[6],   # gripper_command_R
        ) 

    def get_info(self):
        return {
            "success": self._state["buttons"].get("A", False),
            "failure": self._state["buttons"].get("B", False),
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }