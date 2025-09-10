import ALSLib.ALSClient
import ALSLib.ALSSensor
from ALSLib.ALSHelperLidarLibrary import SerializeToPCLFileContent
from ALSLib.ALSSensorDistributionInfo import SensorDistributionInfo
from ALSLib.ALSSensorGroup import SensorGroup
from ALSLib.ALSSensor import Sensor
import os, time, threading, socket, json, inspect, math, struct, copy, cv2, random, queue, numpy as np
import logging
import logging.handlers
import traceback
from typing import Type, List
from enum import Enum
from os import makedirs as osmakedirs, path as ospath, sep as ospath_sep, remove as removefile, listdir
from PIL import Image
import ALSLib.ALSHelperFunctionLibrary as ALSFunc
from ALSLib import TCPClient, ALSClient
from abc import ABC, abstractmethod
import av.datasets, ctypes
from av.container import InputContainer, OutputContainer
from av.stream import Stream
from av.video.stream import VideoStream
from av.packet import Packet

if "ALS_DATACOLLECTION_BASE_DIR" not in os.environ:
	BASE_DIR = ALSFunc.get_sensordata_path()
else:
	BASE_DIR = os.environ["ALS_DATACOLLECTION_BASE_DIR"]

UINT8_SIZE = 1
FSIZE = 4
ISIZE = 4
LID_PT_N = 4 # Number of fields in a lidar data point
LID_RAW_SIZE_H = 11 * FSIZE # Lidar data header size
LID_RAW_SIZE_D = LID_PT_N * FSIZE # Lidar Raw data point size
LID_ANGLE_DISTANCE_SIZE_H = 8 * FSIZE + 4 * ISIZE
OUT_DIR_DEFAULT = "UnnamedDataset"
DEBUG_DISABLE_SAVING = False

def read_str_from_mem_view(buffer, offset):
	# same as ALSHelperFunctionLibrary.ReadString but this one works with memory views
	str_val_len, offset = ALSFunc.ReadUint32(buffer, offset)
	str_val_view = buffer[offset:offset + str_val_len]
	str_val = str()
	try:
		str_val = str(str_val_view, 'utf8')
	except Exception:
		err_str = "error string could not be generated"
		try:
			err_str = str(buffer[offset:offset+max(str_val_len,40)], 'utf8', "ignore")
		except Exception as exc:
			raise Exception(f"Failed to decode string: '{err_str}'") from exc
		#return (str(), offset)
	return (str_val, offset + str_val_len)

def json_from_string(extra_str):
	if not extra_str:
		return None
	try:
		j = json.loads(extra_str)
	except Exception:
		return None
	return j

def load_override_smart(override, alsclient:ALSClient.Client, timeout:float=20):
	config_file_type = ""
	config_file_name = ""
	override_text = ""
	if isinstance(override, str):
		# using raw override string
		override_text = override.replace('\\','/')
		override_path = override_text.split(";")[0]
		config_file_type = override_path.rpartition("/")[0] .lower()
		config_file_name = override_path.split("/")[-1].split(".")[0]

	elif isinstance(override, tuple):
		# using ("type", "filename", "override") tuple
		config_file_type = override[0].strip().lower()
		config_file_name = override[1]
		override_text = override[2]
	else:
		return False

	a = config_file_name.strip()
	b = override_text.strip()

	if "weather" in config_file_type:
		alsclient.request_load_weather_with_overrides(a,b, timeout)
	elif "situation" in config_file_type:
		alsclient.request_load_situation_with_overrides(a,b, timeout)
	elif "seastate" in config_file_type:
		alsclient.request_load_sea_state_with_overrides(a,b, timeout)
	elif "sensors/camerasettings" in config_file_type:
		alsclient.request(f"ReloadPostProcessWithOverrides {b}", timeout)
	else:
		return False

	return True

# Specializations for different kinds of data are defined in subclasses
class SensorData(ABC):
	sensor_type_str = "UNDEFINED_SENSOR_TYPE"
	def __init__(self):
		self.data = None
		self.meta:dict = None
		self.addl_meta:dict = None
		self.loaded_buffer_size:int = 0 # used for offset into a sensor group buffer
		self.ingroup:bool = False
		self.timestamp:float = 0.0
		self.output_path:str = str()
		self.save_meta:bool = False

	class SensorDataException(Exception):
		def __init__(self, caller, message):
			super().__init__(f"{type(caller)}.{inspect.stack()[2][3]}: {message}")

	class VirtualMethodNotImplementedException(SensorDataException):
		def __init__(self, caller, name):
			super().__init__(caller, f"Virtual method {name} not implemented")

	class FileOverwriteException(SensorDataException):
		def __init__(self, caller, full_path:str):
			super().__init__(caller, f"File '{full_path}' already exists")

	class MetadataSaveException(SensorDataException):
		def __init__(self, caller, full_path:str):
			super().__init__(caller, f"Failed to save metadata file '{ospath.split(full_path)[1]}'")

	@abstractmethod
	def load(self, buffer:memoryview, offset:int=0):
		raise SensorData.VirtualMethodNotImplementedException(self, "load")

	@abstractmethod
	def save(self, dir:str, filename:str):
		raise SensorData.VirtualMethodNotImplementedException(self, "save")

	def valid(self):
			return self.data is not None and self.loaded_buffer_size > 0

	def set(self, new_data, bytes_loaded_from_buffer:int):
		self.data = new_data
		self.loaded_buffer_size = bytes_loaded_from_buffer

	def overwrite_check(self, full_path:str) -> bool:
		if ospath.isfile(full_path):
			raise SensorData.FileOverwriteException(self, full_path)
		else:
			return True

	def save_metadata(self, filepath:str):
		if self.meta and self.save_meta:
			try:
				json_data = {}
				if isinstance(self.meta, dict):
					json_data['received_metadata'] = self.meta
				if self.addl_meta and isinstance(self.addl_meta, dict):
					json_data['additional_metadata'] = self.addl_meta
				with open(filepath, "w", encoding="utf-8") as f:
					json.dump(json_data, f, indent=4)
			except Exception:
				raise SensorData.MetadataSaveException(self, filepath)
			
	@classmethod
	@abstractmethod
	def create(cls, config: Sensor) -> None:
		pass

class CameraData(SensorData):
	sensor_type_str = "CAM"
	def __init__(self):
		SensorData.__init__(self)
		self.format:ALSFunc.SensorImageFormat = None

	def load(self, buffer:memoryview, offset:int=0):
		i = offset
		# load image
		if not self.ingroup:
			image, i, w, h, format = ALSFunc.ReadImage_Stream(buffer, i)
		else:
			image, i, w, h, format = ALSFunc.ReadImage_Group(buffer, i)
		self.format = format

		# load metadata (extra string)
		if i < len(buffer):
			extra_str, i = read_str_from_mem_view(buffer, i)
			self.meta = json_from_string(extra_str)
			if self.meta:
				if "T" in self.meta.keys():
					self.timestamp = float(self.meta['T'])
			else:
				self.timestamp = 9999.9999

			if self.format == ALSFunc.SensorImageFormat.Raw:
				b, g, r, a = Image.fromarray((image).astype(np.uint8)).split()
				image = Image.merge("RGB", (r, g, b))

		self.set(image, i - offset)

	def save(self, dir:str, filename:str):
		full_path = ospath.join(dir, filename + ".png")

		if self.overwrite_check(full_path):
			filename = ospath.basename(full_path).removesuffix(".png")
			if self.format == ALSFunc.SensorImageFormat.Raw:
				image_data = np.array(self.data)
				image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
				cv2.imwrite(full_path, image_data, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
			elif self.format == ALSFunc.SensorImageFormat.PNG:
				with open(full_path, 'wb') as file:
					file.write(self.data)
			else:
				print("Unsupported image format\n")
			self.save_metadata(ospath.join(dir, filename + ".json"))
		self.data = None

	@classmethod
	def create(cls, config: Sensor) -> 'CameraData':
		return cls()

class CameraDataRTSPVideo(SensorData):
	# RTSP videos do not require a per-frame data structure, this is only used for differentiating by protocol
	sensor_type_str = "CAM"
	def load(self): pass
	def save(self): pass

	@classmethod
	def create(cls, config: Sensor) -> 'CameraDataRTSPVideo':
		return cls()

class LidarDataRaw(SensorData):
	sensor_type_str = "LID"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer:memoryview, offset:int=0):
		# Read header
		posX, posY, posZ, quatW, quatX, quatY, quatZ, num_points, tStart, tEnd,\
		num_rays = struct.unpack('<fffffffffff', buffer[offset:offset+LID_RAW_SIZE_H])
		# Read data
		self.timestamp = tStart
		data_size = int(num_points * LID_RAW_SIZE_D)
		point_array = np.frombuffer(buffer[offset+LID_RAW_SIZE_H:offset+LID_RAW_SIZE_H+data_size], dtype=np.dtype("float32"))
		point_array = np.reshape(point_array, (-1, LID_PT_N))

		
		# Serialize to Point Cloud Data format (.pcd compatible with cloud compare)
		pcl = SerializeToPCLFileContent(int(num_points), posX, posY, posZ, quatW, quatX, quatY, quatZ, point_array)
		self.set(pcl, LID_RAW_SIZE_H + data_size)

	def save(self, dir:str, filename:str):
		full_path = ospath.join(dir, filename + ".pcd")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

	@classmethod
	def create(cls, config: Sensor) -> 'LidarDataRaw':
		return cls()

class LidarDataAngleDistance(SensorData):
	sensor_type_str = "LID"
	def __init__(self, span_width_degree, span_width_step, span_height_degree):
		SensorData.__init__(self)
		self.span_width_degree = span_width_degree
		self.span_width_step = span_width_step
		self.span_height_degree = span_height_degree

	# r is the distance of the reading
	# theta is angle between Z and r  (!!not (x.y) and r)
	# phi is the angle in the plane (xy)
	def reading_to_3d(self, r,theta,phi, intensity):
		theta_r = np.deg2rad(theta)
		phi_r = np.deg2rad(phi)
		x = r * math.sin(theta_r) * math.cos(phi_r)
		y = r * math.sin(theta_r) * math.sin(phi_r)
		z = r * math.cos(theta_r)
		return [x,y,z,intensity]

	def load(self, buffer:memoryview, offset:int=0):
		# Read header
		point_array = []
		pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,\
		number_of_columns_per_chunk, num_beams_per_column, column_time, FrameID,\
		ColId = struct.unpack('<fffffffiifii', buffer[offset:offset+LID_ANGLE_DISTANCE_SIZE_H])

		# Read data
		num_points_in_frame = 0
		if num_beams_per_column != 0:
			h_steps = int(self.span_width_degree) / int(self.span_width_step)
			v_steps = -int(self.span_height_degree) / num_beams_per_column

			pointcloud_data = buffer[offset+LID_ANGLE_DISTANCE_SIZE_H:]
			readings_array = np.frombuffer(pointcloud_data, dtype=np.dtype("float32"))

			for count, d in enumerate(readings_array):
				reading = self.reading_to_3d(d, 90-(v_steps*(count-num_beams_per_column*0.5)), 180-h_steps*ColId, readings_array[count+num_beams_per_column])
				reading = [reading[0], reading[1], reading[2], reading[3]]
				point_array.append(reading)
				num_points_in_frame += 1
				if count >= num_beams_per_column-1: # after the list of distances we get the list of intensities
					break

		# Serialize to Point Cloud Data format (.pcd compatible with cloud compare)
		pcl = SerializeToPCLFileContent(int(num_points_in_frame), pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, point_array)
		self.set(pcl, LID_ANGLE_DISTANCE_SIZE_H)

	def save(self, dir:str, filename:str):
		full_path = ospath.join(dir, filename + ".pcd")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

	@classmethod
	def create(cls, config: Sensor) -> 'LidarDataAngleDistance':
		span_width_degree = int(config.sensor_info.get('_SpanWidthDegree'))
		if span_width_degree is None:
			raise ValueError("Missing '_SpanWidthDegree' in context")
		span_width_step = int(config.sensor_info.get('_SpanWidthStep'))
		if span_width_step is None:
			raise ValueError("Missing '_SpanWidthStep' in context")
		span_height_degree = int(config.sensor_info.get('_SpanHeightDegree'))
		if span_height_degree is None:
			raise ValueError("Missing '_SpanHeightDegree' in context")
		return cls(span_width_degree, span_width_step, span_height_degree)


class LidarDataLivox(SensorData):
	sensor_type_str = "LID"
	def __init__(self, stream_transform_not_quat:bool):
		SensorData.__init__(self)
		self.stream_transform_not_quat:bool = stream_transform_not_quat
		if self.stream_transform_not_quat:
			self.lid_livox_size_h = 11 * FSIZE
		else:
			self.lid_livox_size_h = 9 * FSIZE
		self.point_size = FSIZE * 5 + (2 * UINT8_SIZE)

	def ___read_data(self, num_points:int, t_start:float, buffer:memoryview, offset:int):
		self.timestamp = t_start
		read_points = 0
		data_size = int(num_points * self.point_size)
		livoxPoints = []
		while read_points < num_points:
			start = offset + self.lid_livox_size_h + read_points * self.point_size
			end = offset + self.lid_livox_size_h + (read_points + 1) * self.point_size
			offsetTime, pointX, pointY, pointZ, intensity, tag, number = struct.unpack('fffffBB', buffer[start:end])
			livoxPoints.append((offsetTime, pointX, pointY, pointZ, intensity, tag, number))
			read_points += 1
		return livoxPoints

	def load(self, buffer:memoryview, offset:int=0):
		# Read header
		
		if self.stream_transform_not_quat:
			pos_x, pos_y, pos_z, pitch, yaw, roll, scale_x, scale_y, scale_z, t_start, num_points = struct.unpack('<fffffffffff', buffer[offset:offset+self.lid_livox_size_h])
			data_size = int(num_points * self.point_size)
			livox_points = self.___read_data(num_points, t_start, buffer, offset)

			# Forming the pointcloud data from livox data
			point_array = [(x[1],x[2],x[3],x[4]) for x in livox_points]
			point_array = np.reshape(point_array, (-1,4))

			txt_file_content = f"{pos_x:.5f} {pos_y:.5f} {pos_z:.5f} {pitch:.5f} {yaw:.5f} {roll:.5f} {scale_x:.5f} {scale_y:.5f} {scale_z:.5f} {t_start:.5f} {num_points:.5f}\n"
			for p in point_array:
				intensity = 1000
				if not math.isinf(p[3]) and not math.isnan(p[3]) and p[3] is not None:
					intensity = int(p[3])
				txt_file_content += f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {intensity:d}\n"
			self.set(txt_file_content, self.lid_livox_size_h + data_size)

		else:
			pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, t_start, num_points = struct.unpack('<fffffffff', buffer[offset:offset+self.lid_livox_size_h])
			data_size = int(num_points * self.point_size)
			livox_points = self.___read_data(num_points, t_start, buffer, offset)
			
			# Forming the pointcloud data from livox data
			point_array = [(x[1],x[2],x[3],x[4]) for x in livox_points]
			point_array = np.reshape(point_array, (-1,4))

			pcl = SerializeToPCLFileContent(int(num_points), pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, point_array)
			self.set(pcl, self.lid_livox_size_h + data_size)
	
	def save(self, dir:str, filename:str):
		ending = ".txt" if self.stream_transform_not_quat else ".pcd"
		full_path = ospath.join(dir, filename + ending)
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

	@classmethod
	def create(cls, config: Sensor) -> 'LidarDataLivox':
		stream_transform_not_quat:bool = bool(config.sensor_info.get("_StreamTransformNotQuat"))
		if stream_transform_not_quat is None:
			stream_transform_not_quat = False
		return cls(stream_transform_not_quat)
	

class LidarDataPCAP(SensorData):
	sensor_type_str = "LID"
	def __init__(self, addr, port, output_simulation_time, save_ground_truth_as_color):
		SensorData.__init__(self)
		self.header_size = 0
		self.addr = addr
		if self.addr == "localhost":
			self.addr = "127.0.0.1"
		self.port = port
		self.output_simulation_time = output_simulation_time
		self.save_ground_truth_as_color = save_ground_truth_as_color
		self.number_of_packets = 0
		self.number_of_points_per_packet:list = []
		type_list = [
			('timestamp', 'uint16'),
			('distance', 'uint16'),
			('azimuth', 'int16'),
			('elevation', 'int16'),
			('reflectivity', 'uint16'),
			('filter_index', 'uint16')
		]
		self.unpack_instruction = '<HHhhHH'
		if self.output_simulation_time:
			type_list.append(('simulation_time', 'float32'))
			self.unpack_instruction.join("f")
		if self.save_ground_truth_as_color:
			type_list.append(('ground_truth', 'float32'))
			self.unpack_instruction.join("f")
		self.point_dtype = np.dtype(type_list)
		
	PCAP_GLOBAL_HEADER = struct.pack(
		'<IHHiIII',
		0xa1b2c3d4,  # Magic number
		2,           # Major version
		4,           # Minor version
		0,           # Time zone (GMT)
		0,           # Accuracy of timestamps
		65535,       # Max length of packets
		1            # Data link type (Ethernet or user-defined)
	)

	def load_single(self, buffer, offset=0) -> tuple[bytes, int]:
		cursor = offset
		self.number_of_packets = 1
		buffer_size = len(buffer)
		header_size = 0
		element_size = self.point_dtype.itemsize
		if (buffer_size - header_size) % element_size != 0:
			raise ValueError("Buffer size must be a multiple of element size after accounting for the header")
		point_data = buffer[header_size:]
		point_array = np.frombuffer(point_data, dtype=self.point_dtype)
		cursor += len(point_array) * element_size + header_size
		self.number_of_points_per_packet.append(len(point_array))
		# Serialize points into the payload
		payload = b""
		for point in point_array:
			payload += self.__pack_point(point)
		return payload, cursor - offset

	def load_group(self, buffer, offset=0) -> tuple[bytes, int]:
		cursor = offset
		element_size = self.point_dtype.itemsize

		payload = b""
		self.number_of_packets, cursor = ALSFunc.ReadUint32(buffer, cursor)
		for current_packet in range(self.number_of_packets):
			number_of_points_in_packet, cursor = ALSFunc.ReadUint32(buffer, cursor)
			start = cursor + self.header_size
			end = start + element_size * number_of_points_in_packet
			point_data = buffer[start:end]
			point_array = np.frombuffer(point_data, dtype=self.point_dtype)
			for point in point_array:
				payload += self.__pack_point(point)
			cursor += self.header_size + number_of_points_in_packet * element_size
			self.number_of_points_per_packet.append(number_of_points_in_packet)
		return payload, cursor - offset


	def load(self, buffer, offset=0) -> None:
		if self.ingroup is True:
			payload, offset = self.load_group(buffer, offset)
		else:
			payload, offset = self.load_single(buffer, offset)
		self.set(payload, offset)

	def save(self, dir, filename):
		# File path for the PCAP file
		full_path = ospath.join(dir, filename + ".pcap")
		# Write global header if the file doesn't exist
		if not ospath.exists(full_path):
			self.__write_global_header(full_path)
		# Write the current packets
		if self.data:
			for i in range(self.number_of_packets):
				#Start is total number of points before current packet
				start = sum(self.number_of_points_per_packet[:i]) * self.point_dtype.itemsize
				end = start + self.number_of_points_per_packet[i] * self.point_dtype.itemsize
				self.__write_packet(full_path, self.data[start:end])
			self.data = None

	def __write_global_header(self, filepath):
		with open(filepath, 'wb') as f:
			f.write(self.PCAP_GLOBAL_HEADER)

	def __write_packet(self, filepath, packet_data) -> None:
		current_time = time.time()
		ts_sec = int(current_time)
		ts_usec = int((current_time - ts_sec) * 1e6)

		# Ethernet Header
		eth_dst = b'\x00\x00\x00\x00\x00\x00' 			# Placeholder
		eth_src = eth_src = b'\x00\x00\x00\x00\x00\x00'	# Placeholder
		eth_type = 0x0800

		# IP Header
		ip_version_ihl = 0x45													# IPv4 4 + IHL (5 * 4 = 20 bytes)
		ip_dscp_ecn = 0x00														# DSCP/ECN
		ip_total_len = 0x00														# Total length
		ip_id = 0x00															# Identification
		ip_flags_frag = 0x00													# Flags and fragment offset
		ip_ttl = 0x80															# TTL (Time to live)
		ip_protocol = 0x11														# Protocol (UDP)
		ip_csum = 0x00															# Checksum (can be calculated)
		ip_src = socket.inet_aton(self.addr)									# Source IP address
		ip_dst = socket.inet_aton(socket.gethostbyname(socket.gethostname()))	# Destination IP address

		# UDP Header
		udp_src_port = struct.unpack('!H', struct.pack('H', int(self.port)))[0]
		udp_dst_port = 0xC000
		udp_len = len(packet_data) + 8
		udp_csum = 0x0000
		
		#Length calculation
		eth_header_len = 14
		ip_header_len = 20
		udp_header_len = 8
		total_len = eth_header_len + ip_header_len + udp_header_len + udp_len
		ip_total_len = total_len

		# Ethernet, IP, and UDP headers struct
		custom_packet_header = struct.pack(
			'6s6sH' +			# Ethernet Header
			'BBHHHBBH4s4s' +	# IP Header
			'HHHH',				# UDP Header
			
			eth_dst,	# mac_dst (6 bytes)
			eth_src,	# mac_src (6 bytes)
			eth_type,	# ether_type (2 bytes)
			
			ip_version_ihl,	# version_ihl (1 byte)
			ip_dscp_ecn,	# dscp_ecn (1 byte)
			ip_total_len,	# total_len (2 bytes)
			ip_id,			# id (2 bytes)
			ip_flags_frag,	# flag_frag (2 bytes)
			ip_ttl,			# ttl (1 byte)
			ip_protocol,	# protocol (1 byte)
			ip_csum,		# csum (2 bytes)
			ip_src,			# ip_src (4 bytes)
			ip_dst,			# ip_dst (4 bytes)
			
			udp_src_port,	# src_port (2 bytes)
			udp_dst_port,	# dst_port (2 bytes)
			udp_len,		# len (2 bytes)
			udp_csum,		# csum (2 bytes)
		)

		packet_size = len(custom_packet_header) + len(packet_data)

		# PCAP Header struct
		pcap_packet_header = struct.pack(
			'<IIII',
			ts_sec,			# Timestamp (seconds)
			ts_usec,		# Timestamp (microseconds)
			packet_size,	# Captured length (size of the captured packet)
			packet_size		# Actual length (total size of the packet)
		)

		# Write
		with open(filepath, 'ab') as f:
			f.write(pcap_packet_header)
			f.write(custom_packet_header)
			f.write(packet_data)

	def __pack_point(self, point):
			values = [
				point['timestamp'],
				point['distance'],
				point['azimuth'],
				point['elevation'],
				point['reflectivity'],
				point['filter_index']
			]
			if self.output_simulation_time:
				values.append(point['simulation_time'])
			if self.save_ground_truth_as_color:
				values.append(point['ground_truth'])
			return struct.pack(self.unpack_instruction, *values)

	@classmethod
	def create(cls, config: Sensor) -> 'LidarDataPCAP':
		address = config.sensor_ip
		if address is None:
			raise ValueError("Missing 'sensor_ip' in sensor distribution information")
		port = config.sensor_port
		if port is None:
			raise ValueError("Missing 'sensor_port' in sensor distribution information")
		output_simulation_time:bool = bool(config.sensor_info.get('_OutputSimulationTime'))
		if output_simulation_time is None:
			output_simulation_time = False
		ground_truth:bool = bool(config.sensor_info.get('_saveGroundTruthAsColor'))
		if ground_truth is None:
			ground_truth = False
		return cls(address, port, output_simulation_time, ground_truth)

class RadarData(SensorData):
	sensor_type_str = "RAD"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer:memoryview, offset:int=0):
		buffer_slice:memoryview = buffer[offset:]
		data:str = None
		try:
			data = str(buffer_slice, 'utf8')
		except Exception as e:
			data = str(buffer_slice[:min(40,len(buffer_slice))], 'utf8', "ignore")
			data = f"{e}: Failed to decode radar data. Partial data string: '{data}'"
			log_to_file(data)

		data_json = None
		try:
			data_json = json.loads(data)
		except json.decoder.JSONDecodeError:
			data_json = None
		if not data_json:
			self.set(data, len(data.encode('utf-8')))
			return

		for i, p in enumerate(data_json["data"]):
			# [ altitude, azimuth, depth, velocity, intensity ] -> [ velocity, azimuth, altitude, depth, intensity ]
			data_json["data"][i] = [ float(p[3]), float(p[1]), float(p[0]), float(p[2]), float(p[4]) ]

		json_str = json.dumps(data_json)
		self.set(json_str, len(json_str.encode('utf-8')))

	def save(self, dir:str, filename:str):
		full_path = ospath.join(dir, filename + ".txt")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None

	@classmethod
	def create(cls, config: Sensor) -> 'RadarData':
		return cls()

class TextData(SensorData):
	sensor_type_str = "TEXT_DATA_BASE_CLASS"
	def __init__(self):
		SensorData.__init__(self)

	def load(self, buffer:memoryview, offset:int=0):
		reading, new_offset = read_str_from_mem_view(buffer, offset)
		self.set(reading, new_offset - offset)

	def save(self, dir:str, filename:str):
		full_path = ospath.join(dir, filename + ".txt")
		if self.overwrite_check(full_path):
			with open(full_path, mode='w', encoding='utf-8') as f:
				f.write(self.data)
		self.data = None
	
	@classmethod
	def create(cls, config: Sensor) -> 'TextData':
		return cls()

class SpeedometerData(TextData):
	sensor_type_str = "SPE"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'SpeedometerData':
		return cls()

class LaserData(TextData):
	sensor_type_str = "LAS"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'LaserData':
		return cls()

class GNSSData(TextData):
	sensor_type_str = "GNS"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'GNSSData':
		return cls()

class SplineData(TextData):
	sensor_type_str = "SPL"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'SplineData':
		return cls()

class IMUData(TextData):
	sensor_type_str = "IMU"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'IMUData':
		return cls()

class FilteredObjectGetterData(TextData):
	sensor_type_str = "FOG"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'FilteredObjectGetterData':
		return cls()

class AISData(TextData):
	sensor_type_str = "AIS"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'AISData':
		return cls()

#ARPA still not working.
class ARPAData(SensorData):
	sensor_type_str = "ARP"
	def __init__(self):SensorData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'ARPAData':
		return cls()

class OccupancyGridData(TextData):
	sensor_type_str = "OCC"
	def __init__(self):TextData.__init__(self)
	def load(self, buffer:memoryview, offset:int=0):TextData.load(self, buffer, offset)
	def save(self, dir:str, filename:str):TextData.save(self, dir, filename)

	@classmethod
	def create(cls, config: Sensor) -> 'OccupancyGridData':
		return cls()

class SensorDataClasses():
	def __init__(self):
		# Default classes, can be changed by the user (per-instance)
		self.camera = CameraData
		self.lidar_raw = LidarDataRaw
		self.lidar_livox = LidarDataLivox
		self.lidar_pcap = LidarDataPCAP
		self.lidar_angle_distance = LidarDataAngleDistance
		self.radar = RadarData
		self.speedometer = SpeedometerData
		self.laser = LaserData
		self.gnss = GNSSData
		self.spline = SplineData
		self.imu = IMUData
		self.fog = FilteredObjectGetterData
		self.ais = AISData
		self.rtsp = CameraDataRTSPVideo
		self.arpa = ARPAData
		self.occupancy_grid = OccupancyGridData

	def select_class(self, a:str, b:str, output_type:str=None, protocol:str=None):
		if a == "CAM" or b == "CAM":
			if protocol != "RTSP":
				return self.camera, "CAM"
			else:
				return self.rtsp, "CAM"
		elif a == "LID" or b == "LID":
			if output_type == "Raw":
				return self.lidar_raw, "LID"
			elif output_type == "Livox":
				return self.lidar_livox, "LID"
			elif output_type == "PCAP":
				return self.lidar_pcap, "LID"
			elif output_type == "AngleDistance":
				return self.lidar_angle_distance, "LID"
			else:
				raise Exception("Error: Found no data class for sensor type {} / {}".format(a,b))
		elif a == "RAD" or b == "RAD":
			return self.radar, "RAD"
		elif a == "SPE" or b == "SPE":
			return self.speedometer, "SPE"
		elif a == "LAS" or b == "LAS":
			return self.laser, "LAS"
		elif a == "GNS" or b == "GNS":
			return self.gnss, "GNS"
		elif a == "SPL" or b == "SPL":
			return self.spline, "SPL"
		elif a == "IMU" or b == "IMU":
			return self.imu, "IMU"
		elif a == "FOG" or b == "FOG":
			return self.fog, "FOG"
		elif a == "AIS" or b == "AIS":
			return self.ais, "AIS"
		elif a == "ARP" or b == "ARP":
			return self.arpa, "ARP"
		elif a == "OCC" or b == "OCC":
			return self.occupancy_grid, "OCC"
		else:
			raise Exception("Error: Found no data class for sensor type {} / {}".format(a,b))

def log_to_file(message:str):
	time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
	log_row = f"{time_str} {message}\n"
	log_path = ospath.join(BASE_DIR, "DataCollectionLog.txt")
	try:
		with open(log_path, 'a+') as logfile:
			logfile.write(log_row)
	except Exception:
		log_path = ospath.join(ospath.abspath(ospath_sep), "DataCollectionLog.txt")
		with open(log_path, 'a+') as logfile:
			logfile.write(log_row)

class SensorOptions():
	def __init__(self):
		self.additional_metadata:dict = None #Add metadata to saved metadata
		self.group_name = str()
		self.alias = [] # Sensor alias
		self.batch_limit = 60 # Number of samples per batch, if we are collecting in batches
		self.burn_in_samples = 0 # Number of samples to discard at start
		self.clear_dir = True # Delete existing files at start
		self.data_class_mappings = SensorDataClasses() # Allows overriding sensor types with a custom class
		self.max_time = 0 # Collection time limit
		self.out_dir = OUT_DIR_DEFAULT # Output directory name
		self.pause_after_collection = True #Pause the manager after collecting batch limit is reached
		self.save_metadata = True # Whether to create metadata files
		self.timeout = 30 # Maximum allowed time between data transmissions
		self.alloc_num = None # DEPRECATED
		self.max_variation_sample_size = None # DEPRECATED

class InvalidOptionsException(Exception): pass

class DataInfo():
	def __init__(self, data_view:memoryview, data_class, filename_format_str:str, options:SensorOptions, is_group_data:bool, i:int, addl_meta:dict, sensor_info:SensorGroup|Sensor):
		self.view:memoryview = data_view
		self.data_class: type[SensorData] = data_class
		self.filename_format_str:str = filename_format_str
		self.is_group_data:bool = is_group_data
		self.frame_index:int = i
		self.addl_meta:dict = addl_meta
		self.output_path:str = options.out_dir
		self.save_meta:bool = options.save_metadata
		self.group_name:str = options.group_name
		self.alias:str = options.alias
		self.data_class_mappings:SensorDataClasses = options.data_class_mappings
		self.sensor_info = sensor_info

class SaveSubthread(threading.Thread):
	def __init__(self, owning_thread, data_info:DataInfo):
		threading.Thread.__init__(self)
		self.owning_thread:SaveThread = owning_thread
		self.data_info:DataInfo = data_info
		self.finished:bool = False
		self.start_time:float = time.time()
		self.duration:float = None

	def run(self):
		if DEBUG_DISABLE_SAVING:
			self.finished = True
			return
		if self.data_info.view == None:
			raise Exception("Failed to save, empty memoryview")
		try:
			if not self.data_info.is_group_data:
				self.save_single(self.data_info.view)
			else:
				self.save_group(self.data_info.view)
		except Exception as e:
			self.owning_thread.subthread_error_message = f"{e}"
			self.owning_thread.error_in_subthread = True
		self.duration = time.time() - self.start_time
		self.finished = True

	def save_single(self, view:memoryview):
		data: SensorData = self.data_info.data_class.create(self.data_info.sensor_info)
		data.ingroup = False
		data.output_path = self.data_info.output_path
		data.save_meta = self.data_info.save_meta
		data.addl_meta = self.data_info.addl_meta
		data.load(view.tobytes(), 0)
		# Use alias set in the sensor info by default
		alias = self.data_info.sensor_info.sensor_alias
		# If alias is set manually, use the first item of the alias list as alias, when saving a single sensor data.
		if type(self.data_info.alias) == list and len(self.data_info.alias) > 0:
			alias = self.data_info.alias[0]
		self.save_to_file(data, str(), alias, data.sensor_type_str)

	def save_group(self, view:memoryview):
		offset = 0
		num_data, offset = ALSFunc.ReadUint32(view, offset)
		try:
			for i in range(num_data):
				sensor_type, offset = read_str_from_mem_view(view, offset)
				sensor_path, offset = read_str_from_mem_view(view, offset)
				output_type = self.data_info.sensor_info.sensors[i].sensor_info.get("_OutputType")
				protocol = self.data_info.sensor_info.sensors[i].protocol
				data_class, sensor_type = self.data_info.data_class_mappings.select_class(sensor_type, sensor_path, output_type, protocol)
				data: SensorData = data_class.create(self.data_info.sensor_info.sensors[i])
				data.ingroup = True
				data.output_path = self.data_info.output_path
				sensor_number = "".join(filter(str.isdigit, sensor_path))
				data.save_meta = self.data_info.save_meta
				data.addl_meta = self.data_info.addl_meta
				# Use alias set in the sensor info by default
				alias = self.data_info.sensor_info.sensors[i].sensor_alias
				# If alias is set manually, use the current item of the alias list as alias, when saving a sensor data.
				if type(self.data_info.alias) == list and len(self.data_info.alias) - 1 >= i:
					alias = self.data_info.alias[i]
				group_name = str()
				if self.data_info.group_name is not None and len(self.data_info.group_name) > 0:
					group_name = self.data_info.group_name
				else:
					group_name = self.data_info.sensor_info.group_name
				data.load(view, offset)
				self.save_to_file(data, group_name, alias, sensor_type, sensor_path, sensor_number)
				offset += data.loaded_buffer_size
		except Exception as e:
			raise Exception(f'{self.data_info.sensor_info.sensors[i]} {e} {traceback.format_exc()}') from e

	def save_to_file(self, data:SensorData, group_name:str, alias_str:str, type_str:str, g_path_str=str(), g_number_str=str()):
		filename = self.data_info.filename_format_str.format(group_name=group_name, alias=alias_str, sensor_type=type_str,\
										frame_id=self.data_info.frame_index, time=data.timestamp, sensor_path=g_path_str, sensor_number=g_number_str)
		if not data.valid():
			raise Exception("Empty data object: {}".format(type(data).__name__))
		path = data.output_path
		if path:
			path = ospath.join(BASE_DIR, path)
			if not ospath.isdir(path):
				osmakedirs(path, exist_ok=True)
		data.save(path, filename)

class PauseResumeException(Exception):
	pass

class LogType(Enum):
	DEBUG = 0
	INFO = 1
	WARNING = 2
	ERROR = 3

class SaveThread(threading.Thread):
	def __init__(self, data_class:type|None, addr:str, port:str, options:SensorOptions, filename_format_str:str, log_queue:queue.Queue, log_level:int, sensor_info:Sensor|SensorGroup):
		threading.Thread.__init__(self)
		self.name:str = f"{port}_{int(str().join( [str(random.randint(1,9)) for digit in [0] * 8] ))}"
		self.addr:str = addr
		self.port:str = port
		self.data_class:type = data_class
		self.filename_format_str:str = filename_format_str
		self.lifetime_sample_count:int = 0
		self.starting_time:float = None
		#Amount of samples collected during a run, reseted when unpaused.
		self.current_sample_count:int = 0
		self.paused:bool = False
		self.terminated:bool = False
		self.stop:bool = False
		self.error_in_subthread:bool = False
		self.subthread_error_message:str = str()
		self.options:SensorOptions = SensorOptions()
		self.subthreads:list[SaveSubthread] = []
		self.execution_times:list[float] = []
		#Amount of samples discarded during collection, reseted when unpaused.
		self.discarded:int = 0
		self.log_queue:queue.Queue = log_queue
		self.logger:logging.Logger = setup_logger(self.log_queue, self.name, log_level)
		self.exception:str|None = None
		self.sensor_info = sensor_info
		
		try:
			self.logger.handlers.clear()
			self.set_options(options)
			self.client = TCPClient.TCPClient(self.addr, self.port, 5)
			self.client.connect(2)
			self.client.set_timeouts(-1, self.options.timeout, self.options.timeout, self.options.timeout)
			self._log(LogType.INFO, f"{self.client.get_alsreceiver_version()}")
		except InvalidOptionsException as e:
			self.exception = f"Invalid options:\n {e}"
			self._log(LogType.ERROR, f"Invalid options on receive process {self.name}")
			self.terminated = True
			return
		except Exception as e:
			self.exception = f"Exception at receive process {self.name}:\n {e}"
			self._log(LogType.ERROR, f"Exception on receive process {self.name}")
			self.terminated = True
			return
		if DEBUG_DISABLE_SAVING:
			self._log(LogType.WARNING, "Saving is disabled")

	def run(self):
		if self.exception:
			return
		self.client.start_receive()
		# Start dedicated main loop for sensor/group
		self.__save_loop()
		self.__free_subthreads(wait=True)
		self.client.stop_receive()
		self.client.disconnect()
		self._log(LogType.INFO, f"Received {self.lifetime_sample_count} samples on {self.addr}:{self.port}")
		self.terminated = True
		self.paused = False

	def __save_loop(self):
		if self.terminated:
			return
		while not self.stop:
			self.__check_pause()
			if not self.paused:
				#Burn in samples by reading the next view, and incrementing the discarded sample amount.
				if self.options.burn_in_samples and self.discarded < self.options.burn_in_samples:
					view = self.__read_view_catch()
					#If we haven't yet received a new view, continue the loop.
					if view is None:
						continue
					self.client.free_view(view)
					self.discarded += 1
					self._log(LogType.INFO, f"Burned in samples {self.discarded}/{self.options.burn_in_samples}")
					continue
				if self.starting_time is None:
					self.starting_time = time.time()
				if not self.stop:
					new_view = self.__read_view_catch()
					if new_view is not None:
						self.subthreads.append(SaveSubthread(self, DataInfo(new_view, self.data_class, self.filename_format_str, self.options, bool(not self.data_class), \
															self.lifetime_sample_count, self.options.additional_metadata, self.sensor_info)))
						self.subthreads[-1].start()
						self.current_sample_count += 1
						self.lifetime_sample_count += 1
						if self.options.batch_limit != 0:
							hours, minutes, seconds = ALSFunc.convert_seconds(ALSFunc.average(self.execution_times) * (self.options.batch_limit + 1 - self.current_sample_count))
							self._log(LogType.INFO, f"Current batch samples: {self.current_sample_count}/{self.options.batch_limit} - Estimated time left: {int(hours)} h {int(minutes)} min {seconds:.2f} s")
						else:
							self._log(LogType.INFO, f"Current sample count: {self.current_sample_count} - Time left: {self.starting_time + self.options.max_time - time.time()}")
			self.__free_subthreads(wait=False)
			if self.error_in_subthread:
				if self.exception is not None and isinstance(self.exception, str):
					self.exception += f"Subthread: {self.subthread_error_message}"
				else:
					self.exception = f"Subthread: {self.subthread_error_message}"
				self.stop = True

	def __read_view_catch(self):
		try:
			return self.client.read_view()
		except Exception as alsreceiver_exception:
			self.exception = f"{alsreceiver_exception}"
			self.stop = True
			return None

	def __free_subthreads(self, wait:bool):
		while len(self.subthreads) > 0:
			if len(self.execution_times) > 5000:
				self.execution_times = self.execution_times[4001:]
			self.execution_times += [thread.duration for thread in self.subthreads if thread.finished and thread.duration is not None]
			# Free memory and remove references of all subthreads that have finished
			[self.client.free_view(thread.data_info.view) for thread in self.subthreads if thread.finished]
			self.subthreads = [thread for thread in self.subthreads if not thread.finished]
			if not wait:
				return
			time.sleep(.05)

	def __check_pause(self):
		if (self.options.batch_limit and self.current_sample_count >= self.options.batch_limit) or \
			(self.options.max_time and self.starting_time is not None and time.time() - self.starting_time >= self.options.max_time):
			self.paused = self.client.pause_receive(True)
			assert(self.paused == True)
			self.current_sample_count = 0
			self._log(LogType.INFO, f"Current total samples: {self.lifetime_sample_count}")
			self.starting_time = None

	def unpause_thread(self):
		assert(self.paused)
		self.discarded = 0
		self.current_sample_count = 0
		self.paused = self.client.pause_receive(False)
		assert(self.paused is False)

	def set_options(self, options:SensorOptions):
		try:
			new_options = copy.copy(self.__sanitize_sensor_options(options))
		except Exception as e:
			self.exception = e

		defaults = SensorOptions()
		# These options will not be reset to their default values
		if new_options.additional_metadata == defaults.additional_metadata:
			new_options.additional_metadata = self.options.additional_metadata
		if new_options.group_name == defaults.group_name:
			new_options.group_name = self.options.group_name
		if new_options.alias == defaults.alias:
			new_options.alias = self.options.alias
		if new_options.out_dir == defaults.out_dir:
			new_options.out_dir = self.options.out_dir
		self.options = new_options

	def __sanitize_sensor_options(self, options:SensorOptions):
		defaults = SensorOptions()
		options.burn_in_samples = int(options.burn_in_samples)
		options.max_time = float(options.max_time)
		if options.max_variation_sample_size != defaults.max_variation_sample_size:
			self._log(LogType.WARNING, "The option 'max_variation_sample_size' is deprecated, please remove it")
		if options.alloc_num != defaults.alloc_num:
			self._log(LogType.WARNING, "The option 'alloc_num' is deprecated, please remove it")
		options.out_dir = str(options.out_dir)
		options.clear_dir = bool(options.clear_dir)
		options.save_metadata = bool(options.save_metadata)
		options.timeout = float(options.timeout)
		if type(options.alias) != list:
			raise InvalidOptionsException(f"{SensorOptions} {options.alias.__name__} was not a list.")
		for alias in options.alias:
			if not isinstance(alias, str):
				if alias != '':
					alias = str(alias)
		if not isinstance(options.data_class_mappings, SensorDataClasses):
			raise InvalidOptionsException(
				f"{type(options.data_class_mappings)} does not inherit from {SensorDataClasses}")
		options.batch_limit = int(options.batch_limit)
		if options.batch_limit == 0 and options.max_time == 0:
			raise Exception("Both batch_limit and max_time cannot be 0")
		elif options.batch_limit > 0 and options.max_time > 0:
				self._log(LogType.INFO, "Both batch_limit and max_time are set - collection will stop once either is reached")
		options.pause_after_collection = bool(options.pause_after_collection)
		if options.additional_metadata is not None and not isinstance(options.additional_metadata, dict):
			raise InvalidOptionsException(
				f"{options.additional_metadata} is not valid dictionary object."
			)
		return options

	def set_format_string(self, format_string:str):
		self.filename_format_str = format_string

	def __stat_bytes_to_mib(stats:dict, key:str) -> dict:
		if key in stats:
			stats[key] = f"{'%.2f'%(float(stats[key]) / 1048576)} MiB"
		return stats

	def get_stats(self) -> dict|None:
		try:
			stats = json.loads(self.client.get_thread_stats())
			SaveThread.__stat_bytes_to_mib(stats, "currentSize")
			SaveThread.__stat_bytes_to_mib(stats, "maxSizeUsed")
			return stats
		except Exception:
			return None

	def get_running_save_operations_num(self) -> int:
		return len([thread for thread in self.subthreads if not thread.finished])

	def _log(self, category:LogType, message:str):
		match category:
			case LogType.DEBUG:
				self.logger.debug(str(message))
			case LogType.INFO:
				self.logger.info(str(message))
			case LogType.WARNING:
				self.logger.warning(str(message))
			case LogType.ERROR:
				self.logger.error(str(message))
			case _:
				self.logger.info(str(message))


class RTSPVideoThread(SaveThread):
	def __init__(self, addr:str, port:str, options:SensorOptions, filename_format_str:str, log_queue:queue.Queue, log_level:int, sensor_info:Sensor):
		threading.Thread.__init__(self)
		self.name:str = f"{port}_{int(str().join( [str(random.randint(1,9)) for digit in [0] * 8] ))}_RTSP"
		self.addr:str = addr if (addr.lower() != "localhost") else "127.0.0.1"
		self.port:str = port
		self.filename_format_str:str = filename_format_str
		self.lifetime_sample_count:int = 0
		#Amount of samples collected during a run, reseted when unpaused.
		self.current_sample_count:int = 0
		self.paused:bool = False
		self.terminated:bool = False
		self.stop:bool = False
		self.options:SensorOptions = SensorOptions()
		#Amount of samples discarded during collection, reseted when unpaused.
		self.discarded:int = 0
		self.exception:int = None
		self.log_queue:queue.Queue = log_queue
		self.logger:logging.Logger = setup_logger(self.log_queue, self.name, log_level)
		self.starting_time:float = None
		self.execution_times:list[float] = []

		self.input_:InputContainer = None
		self.output:OutputContainer = None
		self.in_stream:VideoStream = None
		self.out_stream:Stream = None
		self.meta_cache_max:int = 5000
		self.meta_cache:bytearray = None
		self.meta_cache_size:int = 0
		self.path_with_filename:str = None
		self.do_reload_output:bool = False
		self.supported_containers:list[str] = [".mkv", ".mp4", ".mov", ".avi", ".wmv"]

		self.sensor_info = sensor_info

		try:
			self.logger.handlers.clear()
			self.set_options(options)
		except InvalidOptionsException as e:
			self.exception = f"Invalid options:\n {e}"
			self._log(LogType.ERROR, f"Invalid options on receive process {self.name}")
			self.terminated = True
			return
		except Exception as e:
			self.exception = f"Exception at receive process {self.name}:\n {e}"
			self._log(LogType.ERROR, f"Exception on receive process {self.name}")
			self.terminated = True
			return

	def run(self):
		if self.exception:
			return
		if self.__connect():
			self.__save_loop()
			self.__save_metadata()
			self.__close_stream()
			self._log(LogType.INFO, f"Received {self.lifetime_sample_count} samples on {self.addr}:{self.port}")

		self.terminated = True
		self.paused = False

	def __connect(self) -> bool:
		try:
			self.input_: InputContainer = av.open(
								f"rtsp://{self.addr}:{self.port}/", 'r',
								options={
											'rtsp_transport': 'tcp',
											'stimeout': '5000', # default: 5000000
											'max_delay': '5000',
										},
								timeout=3)
		except Exception as e:
			self.exception = f"RTSP failed to connect: {e}"
			return False

		self.in_stream: VideoStream = self.input_.streams.video[0]
		self.__update_output()

		self.meta_cache = bytearray(self.meta_cache_max)
		return True
	
	def __update_output(self):
		if self.output is not None:
			self.output.close()
		self.path_with_filename = ospath.join(self.__initialize_path(), self.__generate_filename())
		self.output: OutputContainer = av.open(self.path_with_filename, "w")
		self.out_stream: Stream = self.output.add_stream(template=self.in_stream)

	def __save_loop(self):
		packet_id:int = 0
		packet_received_time = time.time()
		for packet in self.input_.demux(self.in_stream):
			packet_id += 1
			if self.stop:
				break
			if self.paused:
				continue
			if self.__check_pause():
				self.__save_metadata()
				continue
			if not packet:
				break
			if self.starting_time is None:
				self.starting_time = time.time()

			if self.options.burn_in_samples and self.discarded < self.options.burn_in_samples:
				self.discarded += 1
				if self.discarded == self.options.burn_in_samples:
					self._log(LogType.INFO, f"{self.name}: Burned in samples {self.discarded}/{self.options.burn_in_samples}")
				continue

			packet.stream = self.out_stream

			metas:list = RTSPVideoThread.__sei_from_packet(packet)
			frames:int = max(1, len(metas))
			self.current_sample_count += frames
			self.lifetime_sample_count += frames

			self.__handle_metadata(metas, { "packet_id":packet_id })
			self.output.mux(packet)

			if len(self.execution_times) > 5000:
				self.execution_times = self.execution_times[4001:]
			self.execution_times.append(time.time() - packet_received_time)
			packet_received_time = time.time()
			if packet_id % 10 == 0:
				self._log_for_packet()

	def __handle_metadata(self, metas:list, extra_fields:dict):
		if not self.options.save_metadata:
			return
		for meta in metas:
			meta_json:dict = None
			if isinstance(meta, dict):
				meta_json = meta
			else:
				key:str = "raw_metadata" if isinstance(meta, str) else "invalid_metadata"
				meta_json = { key:str(meta) }

			if self.options.additional_metadata:
				if isinstance(self.options.additional_metadata, dict):
					meta_json["additional"] = self.options.additional_metadata
				else:
					self._log(LogType.ERROR, "Additional metadata must be of type dict")
			for key in extra_fields:
				meta_json["RTSP"][key] = extra_fields[key]

			meta_bytes = f"{json.dumps(obj=meta_json, indent=4)}\n\n".encode("utf-8")
			meta_size = len(meta_bytes)
			if self.meta_cache_size + meta_size >= self.meta_cache_max:
				self.__save_metadata()
				if meta_size > self.meta_cache_max:
					self.meta_cache_max = meta_size * 2
					self.meta_cache = bytearray(self.meta_cache_max)
			self.meta_cache[self.meta_cache_size:self.meta_cache_size + meta_size] = meta_bytes
			self.meta_cache_size += meta_size

	def __save_metadata(self):
		if self.meta_cache_size > 0:
			video_extension = ospath.splitext(self.path_with_filename)[1].strip().lower()
			metadata_path = self.path_with_filename if (not video_extension) else self.path_with_filename.removesuffix(video_extension)
			with open(f"{metadata_path}.json", "a") as meta_file:
				meta_file.write(bytes(self.meta_cache[:self.meta_cache_size]).decode("utf-8"))
			self.meta_cache_size = 0

	def __sei_from_packet(packet:Packet):
			metas = []
			for frame in packet.decode():
				payload = None
				for data in frame.side_data:
					if data.type == "SEI_UNREGISTERED":
						sei = frame.side_data.get("SEI_UNREGISTERED")
						payload = (ctypes.c_char*sei.buffer_size).from_address(sei.buffer_ptr)
						break
				metadata_json = None
				if payload is not None:
					try:
						metadata_string = bytes(payload).decode("utf-8")
					except Exception:
						pass
					try:
						metadata_json = json.loads(metadata_string)
					except Exception:
						pass
				# Meta will either be: JSON (dict), string, or None 
				metas.append(metadata_json)
			return metas

	def _log_for_packet(self):
		if self.options.batch_limit != 0:
			hours, minutes, seconds = ALSFunc.convert_seconds(ALSFunc.average(self.execution_times) * (self.options.batch_limit + 1 - self.current_sample_count))
			self._log(LogType.INFO, f"Current batch samples: {self.current_sample_count}/{self.options.batch_limit} - Estimated time left: {int(hours)} h {int(minutes)} min {seconds:.2f} s")
		else:
			self._log(LogType.INFO, f"Current sample count: {self.current_sample_count} - Time left: {self.starting_time + self.options.max_time - time.time()}")

	def __check_pause(self):
		if (self.options.batch_limit and self.current_sample_count >= self.options.batch_limit)\
			or (self.options.max_time and self.starting_time and time.time() - self.starting_time >= self.options.max_time):
			self.paused = True
			self.current_sample_count = 0
			self.starting_time = None
			return True
		else:
			return False

	def __generate_filename(self) -> str:
		check_str:str = self.filename_format_str.lower()
		if ("{time}" in check_str) or ("{sensor_path}" in check_str) or ("{sensor_number}" in check_str)\
			or ("{frame_id}" in check_str) or ("{group_name}" in check_str):
			self._log(LogType.WARNING, "One or more unsupported variables in RTSP filename format string (time/frame_id/sensor_path/sensor_number/group_name)")
		alias_str = str()
		if isinstance(self.options.alias, list) and len(self.options.alias) > 0:
			alias_str = self.options.alias[0]
		alias_str = str(self.options.alias) if not isinstance(self.options.alias, list) else str(self.options.alias[0])
		if alias_str is None or len(alias_str) == 0:
			alias_str = str(self.sensor_info.sensor_alias)
		filename = self.filename_format_str.replace("{alias}", alias_str).replace("{sensor_type}", "CAM_RTSP")
		container = ospath.splitext(filename)[1].strip().lower()
		if (not container) or (container not in self.supported_containers):
			return f"{filename}{self.supported_containers[0]}"
		else:
			return filename

	def set_options(self, options:SensorOptions):
		self.do_reload_output = (options.out_dir != self.options.out_dir) or self.do_reload_output
		super().set_options(options)

	def set_format_string(self, format_string:str):
		self.do_reload_output = (format_string != self.filename_format_str) or self.do_reload_output
		super().set_format_string(format_string)

	def __initialize_path(self) -> str:
		full_output_path = ospath.join(BASE_DIR, self.options.out_dir)
		if not ospath.isdir(full_output_path):
			osmakedirs(full_output_path, exist_ok=True)
		return full_output_path

	def __close_stream(self):
		self.input_.close()
		self.output.close()
		self.input_ = None
		self.output = None

	def unpause_thread(self):
		assert(self.paused)
		self.discarded = 0
		self.current_sample_count = 0
		if self.do_reload_output:
			self.__update_output()
		self.paused = False

	def _log(self, category:LogType, message:str):
		super()._log(category, message)

	def get_stats(self) -> dict|None:
		return {}

	def get_running_save_operations_num(self):
		return 0 if (self.terminated or self.paused or self.output is None) else 1


class ManagerState(Enum):
	STOPPED = 1
	RUNNING = 2
	PAUSED = 3

class SensorThreadManager():
	def __init__(self):
		self.threads: list[SaveThread] = []
		self.state = ManagerState.STOPPED
		self.clear_dirs:list[str] = []
		self.log_queue:queue.Queue = queue.Queue()
		self.log_thread:threading.Thread = None
		thread_name:str = "ThreadManager"
		self.thread_log_level:int = logging.INFO
		self.logger:logging.Logger = setup_logger(self.log_queue, thread_name, logging.INFO)

	def add(self, addr:str, port:int, data_class:Type[SensorData], filename:str, options:SensorOptions = SensorOptions(), sensor_info:Sensor = None, als_client:ALSLib.ALSClient.Client = None):
		if self.state != ManagerState.STOPPED:
			raise Exception("Please stop the manager before setting up the sensors.")
		if not issubclass(data_class, SensorData):
			raise Exception(f"Data class doesn't inherit {SensorData} class")
		if not isinstance(options, SensorOptions):
			raise Exception(f"Options is not instance of {SensorOptions}")
		# No sensor info provided, try to get it automatically
		if sensor_info is None:
			if als_client is None:
				raise Exception("No sensor defined or valid client passed as parameter, exiting")
			response = als_client.request_get_sensor_distribution_info(5, True)
			sensor_distribution_info:SensorDistributionInfo = SensorDistributionInfo.from_json(response)
			sensor_info = sensor_distribution_info.get_sensor_with_address_and_port(addr, port)
			if sensor_info is None:
				raise Exception(f"No sensor with in {addr}:{port} available, exiting.")
		if data_class == options.data_class_mappings.rtsp:
			self.__add_rtsp(str(addr), str(port), str(filename), options, sensor_info)
		else:
			self.__add_any(data_class, str(addr), str(port), str(filename), options, sensor_info)

	def add_group(self, addr:str, port:int, filename:str, options:SensorOptions = SensorOptions(), sensor_info:SensorGroup = None, als_client:ALSLib.ALSClient.Client = None):
		if self.state != ManagerState.STOPPED:
			raise Exception("Please stop the manager before setting up the sensors.")
		if not isinstance(options, SensorOptions):
			raise Exception(f"Options is not instance of {SensorOptions}")
		if sensor_info is None:
			if als_client is None:
				raise Exception("No sensor defined or valid client passed as parameter, exiting")
			response = als_client.request_get_sensor_distribution_info(5, True)
			sensor_distribution_info:SensorDistributionInfo = SensorDistributionInfo.from_json(response)
			sensor_info = sensor_distribution_info.get_group_with_address_and_port(addr, port)
			if sensor_info is None:
				raise Exception(f"No group with in {addr}:{port} available, exiting.")
		self.__add_any(None, str(addr), str(port), str(filename), options, sensor_info)

	def __clear_dir(self, dir_path:str):
		full_path = ospath.join(BASE_DIR, dir_path)
		if ospath.isdir(full_path):
			self.logger.info("Clearing directory {}".format(full_path))
			for f in listdir(full_path):
				f = ospath.join(full_path, f)
				if ospath.isfile(f):
					removefile(f)

	def __add_any(self, data_class:type, addr:str, port:str, filename:str, options:SensorOptions, sensor_info:Sensor|SensorGroup = None):
		options_copy = copy.deepcopy(options)
		self.threads.append(SaveThread(data_class, addr, port, options_copy, filename, self.log_queue, self.thread_log_level, sensor_info))
		if options_copy.clear_dir and options_copy.out_dir and not (options_copy.out_dir in self.clear_dirs):
			self.clear_dirs.append(options_copy.out_dir)

	def __add_rtsp(self, addr:str, port:str, filename:str, options:SensorOptions, sensor_info:Sensor):
		options_copy = copy.deepcopy(options)
		self.threads.append(RTSPVideoThread(addr, port, options_copy, filename, self.log_queue, self.thread_log_level, sensor_info))
		if options_copy.clear_dir and options_copy.out_dir and not (options_copy.out_dir in self.clear_dirs):
			self.clear_dirs.append(options_copy.out_dir)

	def start(self):
		self.logger.handlers.clear()
		if self.state == ManagerState.RUNNING:
			return
		if self.state == ManagerState.PAUSED:
			self.logger.warning("Receiver is currently paused, please unpause to continue collection.")
			return
		if not self.threads:
			raise Exception("Error: No sensors added")
		self.start_logger_thread()
		self.logger.info("Starting...")
		self.state = ManagerState.RUNNING

		# Recreate existing threads that were running previously
		for i, thread in enumerate(self.threads):
			if thread.terminated and thread.exception is None:
				new_thread = SaveThread(thread.data_class, thread.addr, thread.port, thread.options, thread.filename_format_str, thread.log_queue, self.thread_log_level, thread.sensor_info)
				self.threads[i] = new_thread

		[self.__clear_dir(d) for d in self.clear_dirs]
		[thread.start() for thread in self.threads]
		self.logger.info(f"Started {len(self.threads)} thread{'s' if len(self.threads) > 1 else ''}")

	def stop(self, wait:bool=True, check_errors:bool=True):
		for thread in self.threads:
			if check_errors and thread.exception:
				raise Exception(str(thread.exception))
			thread.stop = True
		if wait:
			[thread.join() for thread in self.threads]
		self.stop_logger_thread()
		self.state = ManagerState.STOPPED

	def log_enable(self, enable:bool=True):
		self.thread_log_level = logging.INFO if enable else logging.WARNING

	def logger_thread(self, log_queue:queue.Queue):
		root = logging.getLogger()
		handler = logging.StreamHandler()
		color_formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')
		handler.setFormatter(color_formatter)
		root.addHandler(handler)
		root.setLevel(logging.INFO)
		while True:
			try:
				record = log_queue.get()
				if record is None:
					break
				logger = logging.getLogger(record.name)
				logger.handle(record)
			except Exception:
				import sys, traceback
				print('Error in logger thread', file=sys.stderr)
				traceback.print_exc(file=sys.stderr)
		root.removeHandler(handler)

	def start_logger_thread(self):
		if self.log_thread is None:
			self.log_thread = threading.Thread(target=self.logger_thread, args=(self.log_queue,), daemon=True)
			self.log_thread.start()

	def stop_logger_thread(self):
		if self.log_thread is not None:
			self.log_queue.put(None)
			self.log_thread.join()
			self.log_thread = None

	def log_stats(self):
		stats = { "state":f"{self.state}" }
		for i, thread in enumerate(self.threads):
			thread_stats = thread.get_stats()
			thread_stats = thread_stats if thread_stats else { "e":"No stats" }
			thread_stats["runningSaveOps"] = thread.get_running_save_operations_num()
			stats[f"Thread_{i} {thread.name}"] = thread_stats
		#self.logger.info("\n" + json.dumps(stats, indent=3))

	def unpause(self):
		if not self.state == ManagerState.PAUSED:
			raise PauseResumeException("Tried to resume before all threads were paused")
		[thread.unpause_thread() for thread in self.threads]
		self.state = ManagerState.RUNNING

	def __check_stop_on_error(self):
		errors = [str(thread.exception) for thread in self.threads if (thread.terminated and thread.exception)]
		if errors:
			self.stop(check_errors=False)
			raise Exception(" | ".join(errors))
		return False

	def __check_enter_pause_state(self) -> bool:
		paused_count = 0
		for thread in self.threads:
			paused_count += int(thread.paused)
		if paused_count == len(self.threads):
			self.state = ManagerState.PAUSED
			self.log_stats()
		return self.state == ManagerState.PAUSED

	def is_paused(self) -> bool:
		# to maintain backwards compatibility, the state will only be changed when the user calls is_paused()
		return self.state == ManagerState.PAUSED or self.__check_enter_pause_state() or self.__check_stop_on_error()

	def stop_if_finished(self):
		return self.state != ManagerState.STOPPED

	def set_options(self, options:SensorOptions, port:int=None, format_string:str=None):
		if self.state == ManagerState.RUNNING:
			self.logger.warning("Pausing the receiver before modifying options is recommended")
		if options.clear_dir and options.out_dir and (options.out_dir not in self.clear_dirs):
			self.__clear_dir(ospath.abspath(options.out_dir))
			self.clear_dirs.append(options.out_dir)
		[thread.set_options(options) for thread in self.threads if (port is None or thread.port == str(port))]
		[thread.set_format_string(format_string) for thread in self.threads if (format_string and (port is None or thread.port == str(port)))]

	def auto_setup(self, als_client:ALSClient.Client, options:SensorOptions=SensorOptions(), filename_format_str:str=str()):
		if self.state != ManagerState.STOPPED:
			raise Exception("Please stop the manager before setting up the sensors.")
		self.logger.handlers.clear()
		self.start_logger_thread()
		response = als_client.request_get_sensor_distribution_info(20, True)
		#If we receive error message from the simulator, raise an exception.
		if response.startswith("error "):
			response = response.removeprefix("error ")
			data = json.loads(response)
			raise Exception(data["error"])
		
		sensor_distribution_info:SensorDistributionInfo = SensorDistributionInfo.from_json(response)

		self.threads = []
		self.clear_dirs = []

		active_groups = sensor_distribution_info.get_active_groups()
		active_sensors = sensor_distribution_info.get_active_sensors()

		auto_dir_name = options.out_dir == OUT_DIR_DEFAULT
		num_groups_skip = len(sensor_distribution_info.groups)-len(active_groups)
		if num_groups_skip:
			self.logger.warning("{} group{} not streaming, skipped".format(num_groups_skip, "s" if num_groups_skip>1 else ""))
		if len(active_sensors) == 0:
			self.logger.warning("No streaming sensors found")

		for group in active_groups:
			if group.sensors is None or len(group.sensors) < 1:
				self.logger.warning(f"Group {group.group_name} force-enabled (sensors not found)")
			options_copy = copy.copy(options)
			if auto_dir_name:
				options_copy.out_dir = group.group_name
			if group.group_name:
				options_copy.group_name = group.group_name
			aliases = [sensor.sensor_alias for sensor in group.sensors]
			options_copy.alias = aliases
			if not filename_format_str:
				default_format = "{sensor_type}{sensor_number}_{frame_id}"
				default_format = "{alias}_" + default_format if len(options_copy.alias) > 0 and any(alias for alias in options_copy.alias) else default_format
				default_format = "{group_name}_" + default_format if len(group.group_name) > 0 else default_format
				self.__add_any(None, group.socket_ip, group.socket_port, default_format, options_copy, sensor_info=group)
			else:
				self.__add_any(None, group.socket_ip, group.socket_port, filename_format_str, options_copy, sensor_info=group)
			self.logger.info(f"Added group '{group.group_name}' ({len(group.sensors)} sensors) {group.socket_ip}:{group.socket_port}")

		# Add standalone sensors
		type_counts = {}
		for i, sensor in enumerate(active_sensors):
			type = sensor.sensor_type
			if type in type_counts:
				type_counts[type] += 1
			else:
				type_counts[type] = 1

			sensor_id = "{t}{i}".format(t=type, i=type_counts[type]-1)
			output_type = sensor.sensor_info.get("_OutputType")
			data_class = options.data_class_mappings.select_class(type, type, output_type, sensor.protocol)[0]
			options_copy = copy.copy(options)
			if auto_dir_name:
				options_copy.out_dir = sensor_id
			options_copy.alias = [ sensor.sensor_alias if len(sensor.sensor_alias) > 0 else str() ]
			if not filename_format_str:
				default_format = "{sensor_type}{sensor_number}_{frame_id}" if (data_class \
									is not options.data_class_mappings.rtsp) else "{sensor_type}"
				default_format = "{alias}_" + default_format if len(options_copy.alias) > 0 \
									and any(alias for alias in options_copy.alias) else default_format
				if data_class == options.data_class_mappings.rtsp:
					self.__add_rtsp(sensor.sensor_ip, str(sensor.sensor_port), str(default_format), options_copy, sensor_info=sensor)
				else:
					self.__add_any(data_class, str(sensor.sensor_ip), str(sensor.sensor_port), str(default_format), options_copy, sensor_info=sensor)
			else:
				if data_class == options.data_class_mappings.rtsp:
					self.__add_rtsp(sensor.sensor_ip, str(sensor.sensor_port), str(filename_format_str), options_copy, sensor_info=sensor)
				else:
					self.__add_any(data_class, str(sensor.sensor_ip), str(sensor.sensor_port), str(filename_format_str), options_copy, sensor_info=sensor)

			if len(options_copy.alias[0]) > 0:
				self.logger.info(f"Added sensor '{options_copy.alias[0]}' {sensor_id} {sensor.sensor_ip}:{sensor.sensor_port}")
			else:
				self.logger.info(f"Added sensor {sensor_id} {sensor.sensor_ip}:{sensor.sensor_port}")

	def collect_samples_until_paused(self, logging_enabled:bool=True, delta_time:float=0.05):
		if self.state == ManagerState.STOPPED:
			self.start()
			self.log_enable(logging_enabled)
		elif self.state == ManagerState.PAUSED:
			self.unpause()
		while self.state == ManagerState.RUNNING:
			time.sleep(delta_time)
			self.is_paused()

def setup_logger(log_queue:queue.Queue, logger_id:str, log_level:int=logging.INFO) -> logging.Logger:
		logger = logging.getLogger(logger_id)
		if not any(isinstance(handler, logging.handlers.QueueHandler) for handler in logger.handlers):
			logger.setLevel(log_level)
			queue_handler = logging.handlers.QueueHandler(log_queue)
			logger.addHandler(queue_handler)
		return logger

class ColorFormatter(logging.Formatter):
	COLORS = {
		"DEBUG": "\033[90m",    # Grey
		"INFO": "\033[37m",     # White
		"WARNING": "\033[33m",  # Yellow
		"ERROR": "\033[31m",    # Red
		"CRITICAL": "\033[1;31m"  # Bold Red
	}
	RESET = "\033[0m"

	def format(self, record):
		log_color = self.COLORS.get(record.levelname, self.RESET)
		message = super().format(record)
		return f"{log_color}{message}{self.RESET}"
