import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence, Optional

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_BAUDRATE = 8
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = False

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    new_baudrate: Optional[int] = None
    def __init__(self,
                 ids: Sequence[int],
                port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VPII-if00-port0",
                baudrate: int = 57600,
                gripper_config: Sequence[int] = (1, 0)
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        if gripper_config[1] == 0:
            self._ids = ids 
        else:
            ids = list(ids)
            ids.extend([7, 8])
            self._ids = ids
        self._joint_angles = None
        self._lock = Lock()
        self.leader_dis = gripper_config[0]
        self.follower_dis = gripper_config[1]
        self.new_baudrate = 1e6

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")
        
        if self.new_baudrate is not None:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_BAUDRATE, 3
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(
                        f"Failed to set baudrate for Dynamixel with ID {dxl_id}"
                    )
            if not self._portHandler.setBaudRate(self.new_baudrate):
                raise RuntimeError(f"Failed to change the baudrate, {self.new_baudrate}")                  

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

        if self.follower_dis > 0:
            # self.enable_torque(7,0)
            self._packetHandler.write1ByteTxRx(self._portHandler, 8, 11, 5)
            self.enable_torque(8,1)
            self._packetHandler.write2ByteTxRx(self._portHandler, 8, 102, 200)
            self.leader_init = self.read_position(7)
            self.follow_init = self.read_position(8)
            print(f"leader_init: {self.leader_init}, follow_init: {self.follow_init}")

        self._stop_thread = Event()
        self._start_reading_thread()

        self.counter = 0

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")

        for dxl_id, angle in zip(self._ids, joint_angles):
            # Convert the angle to the appropriate value for the servo
            position_value = int(angle * 2048 / np.pi)

            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._groupSyncWrite.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")

        # Clear syncwrite parameter storage
        self._groupSyncWrite.clearParam()

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                # print(f"dxl_id: {dxl_id}, torque_value: {torque_value}")
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )

        self._torque_enabled = enable

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_angles(self):
        # Continuously read joint angles and update the joint_angles array
        # begin = time.time()
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            with self._lock:
                # start = time.time()
                _joint_angles = np.zeros(len(self._ids), dtype=int)
                dxl_comm_result = self._groupSyncRead.txRxPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"warning, comm failed: {dxl_comm_result}")
                    continue
                for i, dxl_id in enumerate(self._ids):
                    if self._groupSyncRead.isAvailable(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    ):
                        angle = self._groupSyncRead.getData(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        )
                        # angle = np.int32(np.uint32(angle))
                        angle = np.array(angle).astype(np.int32)
                        _joint_angles[i] = angle
                    else:
                        raise RuntimeError(
                            f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
                        )
                self._joint_angles = _joint_angles[:-2]

                if self.follower_dis > 0:
                    # pos_7 = self.read_position(7)
                    # pos_8_now = self.read_position(8)
                    pos_7 = _joint_angles[-2]
                    pos_8_now = _joint_angles[-1]
                    pos_8 = int(self.follow_init + (pos_7 - self.leader_init) * (self.follower_dis / self.leader_dis))
                    if pos_8 < self.follow_init-self.follower_dis:
                        pos_8 = self.follow_init-self.follower_dis
                    elif pos_8 > self.follow_init:
                        pos_8 = self.follow_init

                    self.write_position(8, pos_8)
                    self._gripper_aciton = np.array([pos_8_now, pos_8],dtype=int)
                # self.counter += 1
                # if self.counter % 100 == 0:
                #     print(f"time: {time.time()-begin}")
                # end = time.time()
                # print(f"Time elapsed: {end - start}")
                # self.write_position(8, self.pos_8)
            # self._groupSyncRead.clearParam() # TODO what does this do? should i add it
                
    def read_position(self,dxl_id):
    # 读取当前位置
        dxl_present_position, dxl_comm_result, dxl_error = self._packetHandler.read4ByteTxRx(self._portHandler, dxl_id, ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(dxl_comm_result)
            print(dxl_error)
            raise RuntimeError(f"Failed to read position for Dynamixel with ID {dxl_id}")
        return dxl_present_position

    def write_position(self,dxl_id, position):
    # 写入目标位置
        dxl_comm_result, dxl_error = self._packetHandler.write4ByteTxRx(self._portHandler, dxl_id, ADDR_GOAL_POSITION, position)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(dxl_comm_result)
            print(dxl_error)
            raise RuntimeError(f"Failed to write position for Dynamixel with ID {dxl_id}")

    def enable_torque(self,dxl_id,torque):
    # 启用扭矩
        dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque)
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            print(dxl_comm_result)
            print(dxl_error)
            raise RuntimeError(f"Failed to enable torque for Dynamixel with ID {dxl_id}")

    def get_joints(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        while self._joint_angles is None:
            time.sleep(0.1)
        # with self._lock:
        _j = self._joint_angles.copy()
        # 用弧度制表示关节角度
        return _j / 2048.0 * np.pi

    def get_gripper(self) -> np.ndarray:
        return self._gripper_aciton

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self.enable_torque(8,0)
        if self.new_baudrate is not None:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_BAUDRATE, 1
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(
                        f"Failed to set baudrate for Dynamixel with ID {dxl_id}"
                    )
        self._portHandler.closePort()
        print("Port closed")


def main():
    # Set the port, baudrate, and servo IDs
    ids = [1,2,3,4,5,6]

    # Create a DynamixelDriver instance
    try:
        driver = DynamixelDriver(ids)
    except FileNotFoundError:
        driver = DynamixelDriver(ids, port="/dev/cu.usbserial-FT7WBMUB")

    # Test setting torque mode
    driver.set_torque_mode(True)
    driver.set_torque_mode(False)

    # Test reading the joint angles
    try:
        while True:
            joint_angles = driver.get_joints()
            # print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
    except KeyboardInterrupt:
        driver.close()


if __name__ == "__main__":
    main()  # Test the driver
