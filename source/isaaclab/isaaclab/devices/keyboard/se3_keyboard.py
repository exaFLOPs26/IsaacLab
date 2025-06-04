"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni
import time
import math
from ..device_base import DeviceBase

class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

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

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "K": True,
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }


class Se3Keyboard_BMM(DeviceBase):
    """A keyboard controller for sending SE(3) commands for two arms and 2D commands for a mobile base."""

    def __init__(self, pos_sensitivity: float = 1, rot_sensitivity: float = 1, base_sensitivity: float = 0.3):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling for arms.
            rot_sensitivity: Magnitude of input rotation command scaling for arms.
            base_sensitivity: Magnitude of input command scaling for the mobile base.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_sensitivity = base_sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
       
        # bindings for keyboard to command
        self._create_key_bindings()

        # command buffers
        self._close_gripper_left = False
        self._close_gripper_right = False
        self._delta_pos_left = np.zeros(3)  # (x, y, z) for left arm
        self._delta_rot_left = np.zeros(3)  # (roll, pitch, yaw) for left arm
        self._delta_pos_right = np.zeros(3)  # (x, y, z) for right arm
        self._delta_rot_right = np.zeros(3)  # (roll, pitch, yaw) for right arm
        self._delta_base = np.zeros(3)  # (x, y, yaw) for mobile base
        
        self.arm_number = 0 # right arm for 2k, left arm for 2k+1
        self.bimanual_mode = False # Single mode for 2k, Bimanual arm for 2k+1

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def reset(self):
        """Reset all commands."""
        self._close_gripper_left = False
        self._close_gripper_right = False
        self._delta_pos_left = np.zeros(3)
        self._delta_rot_left = np.zeros(3)
        self._delta_pos_right = np.zeros(3)
        self._delta_rot_right = np.zeros(3)
        self._delta_base = np.zeros(3)

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
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
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose commands for left arm, right arm, and mobile base.
        """
        # convert to rotation vectors
        rot_vec_left = Rotation.from_euler("XYZ", self._delta_rot_left).as_rotvec()
        rot_vec_right = Rotation.from_euler("XYZ", self._delta_rot_right).as_rotvec()
        # return the commands
        return (
            np.concatenate([self._delta_pos_left, rot_vec_left]),  # Left arm
            self._close_gripper_left,  # Left gripper
            np.concatenate([self._delta_pos_right, rot_vec_right]),  # Right arm
            self._close_gripper_right,  # Right gripper
            self._delta_base,  # Mobile base
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to handle keyboard events."""
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # if event.input.name == "Y":
            #     self.reset()
            
            # Toggle right-left arm
            if event.input.name == "Y":
                self.arm_number = 1 - self.arm_number  # Toggle between 0 and 1
                
            # Toggle bimanual mode
            elif event.input.name == "B":
                self.bimanual_mode = 1 - self.bimanual_mode
                
            # Position control
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                if self.arm_number == 0:
                    self._delta_pos_right += self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_pos_left += self._INPUT_KEY_MAPPING[event.input.name]
                
            # Rotation control
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                if self.arm_number == 0:
                    self._delta_rot_right += self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_rot_left += self._INPUT_KEY_MAPPING[event.input.name]
                
            # Right Gripper
            elif event.input.name == "M":
                self._close_gripper_right  = not self._close_gripper_right
            elif event.input.name == "N":
                self._close_gripper_left  = not self._close_gripper_left
                    
            # Mobile base & Bimanual mode
            elif event.input.name in ["U", "O"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right += self._INPUT_KEY_MAPPING[event.input.name] 
                else:
                    print("Base movement")
                    self._delta_base += self._INPUT_KEY_MAPPING[event.input.name]
                    
            elif event.input.name in ["I"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right += self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base += self._INPUT_KEY_MAPPING[event.input.name]
            
            elif event.input.name in ["K"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right += self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base += self._INPUT_KEY_MAPPING[event.input.name]
            
            elif event.input.name in ["J"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right += self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base += self._INPUT_KEY_MAPPING[event.input.name]
            
            elif event.input.name in ["L"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right += self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base += self._INPUT_KEY_MAPPING[event.input.name]
        
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Position control
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                if self.arm_number == 0:
                    self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_pos_left -= self._INPUT_KEY_MAPPING[event.input.name]
                
            # Rotation control
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                if self.arm_number == 0:
                    self._delta_rot_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_rot_left -= self._INPUT_KEY_MAPPING[event.input.name]
            # elif event.input.name in ["I", "K", "J", "L", "U", "O"]:
            #     self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
            # elif event.input.name in ["M", "N", "B", "Y", "P", ";"]:
            #     self._delta_rot_right -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["U", "O"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base -= self._INPUT_KEY_MAPPING[event.input.name]
                    
            elif event.input.name in ["I"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["K"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["J"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["L"]:
                if self.bimanual_mode == 1:
                    self.arm_number = 1
                    self._delta_pos_right -= self._INPUT_KEY_MAPPING[event.input.name]
                else:
                    self._delta_base -= self._INPUT_KEY_MAPPING[event.input.name]
            # 1.25 differ by hardware
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        return True

    def _create_key_bindings(self):
        """Creates default key bindings."""
        self._INPUT_KEY_MAPPING = {
            # Left arm
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            # Right arm
            "I": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "K": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "J": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "L": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,

            # "M": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # "N": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # "B": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            # "Y": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # "P": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            # ";": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,

            # Mobile base
            "U": np.asarray([0.0, 0.0, 0.05]) * self.base_sensitivity,
            "O": np.asarray([0.0, 0.0, -0.05]) * self.base_sensitivity,

        }
