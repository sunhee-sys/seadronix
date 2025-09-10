import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from ALSLib.ALSSensorGroup import SensorGroup
from ALSLib.ALSSensor import Sensor

@dataclass
class SensorDistributionInfo:
	groups: List[SensorGroup]
	sensors: List[Sensor]
	_group_map: Dict[str, SensorGroup] = field(init=False, repr=False)
	_individual_sensor_map: Dict[str, Sensor] = field(init=False, repr=False)

	def __post_init__(self):
		self._group_map = {group.group_name: group for group in self.groups}
		self.__link_sensors_to_groups()
		self._individual_sensor_map = {sensor.path: sensor for sensor in self.sensors if self._group_map.get(sensor.sensor_group) is None}

	def __link_sensors_to_groups(self):
		for sensor in self.sensors:
			group = self._group_map.get(sensor.sensor_group)
			if group:
				group.sensors.append(sensor)

	def __normalize_host(self, host: str) -> str:
		return "127.0.0.1" if host == "localhost" else host

	def get_group_with_name(self, group_name: str) -> SensorGroup | None:
		return self._group_map.get(group_name)
	
	def get_group_with_address_and_port(self, address: str, port: int) -> SensorGroup | None:
		address = self.__normalize_host(address)
		return next((group for group in self.groups if self.__normalize_host(group.socket_ip) == address and group.socket_port == port), None)
	
	def get_active_groups(self) -> List[SensorGroup]:
		active_groups = [ group for group in self._group_map.values() if group.stream_to_network.lower() == "true" ]
		return active_groups
	
	def get_sensor_with_path(self, sensor_path: str) -> Sensor | None:
		return self._individual_sensor_map.get(sensor_path)
	
	def get_sensor_with_address_and_port(self, address: str, port: int) -> Sensor | None:
		address = self.__normalize_host(address)
		return next((sensor for sensor in self.sensors if self.__normalize_host(sensor.sensor_ip) == address and sensor.sensor_port == port), None)
	
	def get_sensors(self) -> List[Sensor]:
		return list(self._individual_sensor_map.values())
	
	def get_active_sensors(self) -> List[Sensor]:
		active_sensors = [ sensor for sensor in self._individual_sensor_map.values() if sensor.stream_to_network.lower() == "true" ]
		return active_sensors
 	
	@classmethod
	def from_dict(cls, data: dict) -> "SensorDistributionInfo":
		groups = [SensorGroup(**g) for g in data.get("groups", [])]
		sensors = [Sensor(**s) for s in data.get("sensors", [])]
		return cls(groups=groups, sensors=sensors)

	@classmethod
	def from_json(cls, json_str: str) -> "SensorDistributionInfo":
		data = json.loads(json_str)
		return cls.from_dict(data)
