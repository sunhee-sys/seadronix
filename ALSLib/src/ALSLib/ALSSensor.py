from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Sensor:
	client_id: int
	path: str
	ego_file: str
	ego_path: str
	ego_alias: str
	sensor_type: str
	sensor_port: int
	protocol: str
	sensor_ip: str
	sensor_group: str
	sensor_alias: str
	stream_to_network: str
	sensor_info: Dict[str, Any]
