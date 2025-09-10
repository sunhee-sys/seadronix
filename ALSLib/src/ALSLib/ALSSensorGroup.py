from ALSLib.ALSSensor import Sensor
from dataclasses import dataclass, field
from typing import List

@dataclass
class SensorGroup:
	client_id: int
	group_name: str
	socket_port: int
	socket_ip: str
	socket_protocol: str
	stream_to_network: str
	sensors: List["Sensor"] = field(default_factory=list)
