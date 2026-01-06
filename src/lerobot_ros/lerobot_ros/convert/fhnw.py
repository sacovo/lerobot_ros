import torch
from fhnw_interfaces.msg import DrivingInputControlMsg, DrivingMotorStatesMsg

from .base import BaseTopic


class DrivingInputControlMsgTopic(BaseTopic):
    @staticmethod
    def msg_type():
        return DrivingInputControlMsg

    def feature_description(self):
        return {
            "dtype": "float32",
            "shape": (5,),
            "names": [
                "curvature",
                "tangential_velocity",
                "crab_angle",
                "zero_offset",
                "speed_stage",
            ],
        }

    def to_tensor(self, msg: DrivingInputControlMsg) -> torch.Tensor:
        """Convert a ROS DrivingInputControlMsg message to a PyTorch tensor."""
        return torch.tensor(
            [
                1 / msg.turning_radius if msg.turning_radius != 0 else 10.0,
                msg.tangential_velocity,
                msg.crab_angle,
                msg.zero_offset,
                msg.speed_stage,
            ],
            dtype=torch.float32,
        )

    def from_tensor(self, tensor: torch.Tensor) -> DrivingInputControlMsg:
        """Convert a PyTorch tensor to a ROS DrivingInputControlMsg message."""
        if tensor.shape != (5,):
            raise ValueError(
                "Tensor must have shape (5,) for DrivingInputControlMsg conversion."
            )
        msg = DrivingInputControlMsg()
        msg.turning_radius = 1 / tensor[0].item()
        msg.tangential_velocity = tensor[1].item()
        msg.crab_angle = tensor[2].item()
        msg.zero_offset = tensor[3].item()
        msg.speed_stage = tensor[4].item()
        return msg


class DrivingMotorStatesMsgTopic(BaseTopic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.names = [
            f"{motor}_{i}_{param}"
            for (motor, param, i) in itertools.product(
                ["speed", "angle"],
                ["velocity", "position", "current"],
                range(4),
            )
        ]
        self.names += [
            "speed_status",
            "speed_error",
            "angle_status",
            "angle_error",
        ]

    @staticmethod
    def msg_type():
        return DrivingMotorStatesMsg

    def feature_description(self):
        return {
            "dtype": "float32",
            "shape": (len(self.names),),
            "names": self.names,
        }

    def to_tensor(self, msg):
        values = []
        for motor in [msg.speed_controller, msg.angle_controller]:
            values.extend(motor.speeds)
            values.extend(motor.positions)
            values.extend(motor.currents)
        for motor in [msg.speed_controller, msg.angle_controller]:
            values.append(motor.status)
            values.append(motor.error)

        return torch.tensor(values, dtype=torch.float32)
