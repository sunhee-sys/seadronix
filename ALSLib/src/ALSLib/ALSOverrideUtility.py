# IMPORTANT: Keep this file free from third-part imports, as the simulator does not have access to them.
# It has to stay compatible with overide functions which use the simulator's bundled Python interpreter.
import math, copy, ast

class ALSVector:
	def __init__(self, x_in=0.0, y=0.0, z=None, w=None):
		x = copy.copy(x_in)

		if isinstance(x, str):
			x = dict(ast.literal_eval(x))
			assert(isinstance(x, dict)), "ALSVector construct from string requires valid dictionary format"

		if not isinstance(x, dict):
			self.x = x
			self.y = y
			self.z = z
			self.w = w
		else:
			assert(w == None and z == None and y == 0.0), "ALSVector from dict/string cannot take other arguments"
			self.x = x.get("X", 0.0)
			self.y = x.get("Y", 0.0)
			self.z = x.get("Z", None)
			self.w = x.get("W", None)

	def to_dict(self):
		out_dict = { "X": self.x, "Y": self.y }
		if self.z != None:
			out_dict["Z"] = self.z
		if self.w != None:
			out_dict["W"] = self.w
		return out_dict

	def __str__(self):
		return "".join(str(self.to_dict()).split())
	
	def make_from(unknown):
		if isinstance(unknown, ALSVector):
			return unknown
		elif isinstance(unknown, dict) or isinstance(unknown, str):
			return ALSVector(unknown)
		elif isinstance(unknown, int) or isinstance(unknown, float):
			return ALSVector(unknown, unknown, unknown, unknown)

	def make_euler(pitch=0, yaw=0, roll=0):
		return ALSVector(roll, pitch, yaw)

	def to_quaternion(self):
		if (self.w != None):
			return ALSVector(self.x, self.y, self.z, self.w)
		# UE5 TRotator specifies: X=Roll, Y=Pitch, Z=Yaw
		roll = math.radians(self.x)
		pitch = math.radians(self.y)
		yaw = math.radians(self.z)
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		cp = math.cos(pitch * 0.5)
		sp = math.sin(pitch * 0.5)
		cr = math.cos(roll * 0.5)
		sr = math.sin(roll * 0.5)
		# ZYX order
		w = cr * cp * cy + sr * sp * sy
		x = sr * cp * cy - cr * sp * sy
		y = cr * sp * cy + sr * cp * sy
		z = cr * cp * sy - sr * sp * cy
		return ALSVector(x, y, z, w)
	
	def to_euler(self):
		if (self.w == None):
			return ALSVector(self.x, self.y, self.z)
		sin_pitch = 2.0 * (self.w * self.y - self.z * self.x)
		if abs(sin_pitch) >= 1:
			pitch = math.copysign(math.pi / 2, sin_pitch)
		else:
			pitch = math.asin(sin_pitch)
	
		sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
		cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
		roll = math.atan2(sinr_cosp, cosr_cosp)
	
		siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
		cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
		yaw = math.atan2(siny_cosp, cosy_cosp)
	
		return ALSVector(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

	def _get_component(self, index):
		if index == 0:
			return self.x
		elif index == 1:
			return self.y
		elif index == 2:
			return self.z
		else:
			return self.w

	def _set_component(self, index, value):
		if index == 0:
			self.x = value
		elif index == 1:
			self.y = value
		elif index == 2:
			self.z = value
		else:
			self.w = value

	def single_component_math_in_place(self, index, operator, other):
		a = self._get_component(index)
		b = other._get_component(index)
		if (a != None) and (b != None):
			if operator == "+":
				self._set_component(index, a + b)
			elif operator == "-":
				self._set_component(index, a - b)
			elif operator == "*":
				self._set_component(index, a * b)
			elif operator == "/":
				self._set_component(index, a / b)
			else:
				raise ValueError(f"Invalid operator")
			
	def component_math(self, operator, other):
		out_vector = ALSVector.make_from(copy.copy(self))
		other_vector = ALSVector.make_from(other)
		out_vector.single_component_math_in_place(0, operator, other_vector)
		out_vector.single_component_math_in_place(1, operator, other_vector)
		out_vector.single_component_math_in_place(2, operator, other_vector)
		out_vector.single_component_math_in_place(3, operator, other_vector)
		return out_vector

	def __add__(self, other):
		# the z/w components of the left-most vector are used in case the other vector is narrower
		return self.component_math("+", other)
	
	def __radd__(self, other):
		return ALSVector.make_from(other).component_math("+", self)
	
	def __sub__(self, other):
		return self.component_math("-", other)
	
	def __rsub__(self, other):
		return ALSVector.make_from(other).component_math("-", self)
	
	def __mul__(self, other):
		return self.component_math("*", other)
		
	def __rmul__(self, other):
		return ALSVector.make_from(other).component_math("*", self)
	
	def __truediv__(self, other):
		return self.component_math("/", other)
	
	def __rtruediv__(self, other):
		return ALSVector.make_from(other).component_math("/", self)

	def normalized(self):
		magnitude = math.sqrt(self.x * self.x + self.y * self.y + ((self.z * self.z) if (self.z != None) else 0) + ((self.w * self.w) if (self.w != None) else 0))
		return ALSVector(self.x / magnitude, self.y / magnitude, self.z / magnitude)
	
def MakeALSVector3(x=0, y=0, z=0):
	return ALSVector(x, y, z)

def MakeALSVector4(x=0, y=0, z=0, w=0):
	return ALSVector(x, y, z, w)

class ALSTransform:
	def __init__(self, location_in:ALSVector=MakeALSVector3(), rotation:ALSVector=MakeALSVector4(), scale:ALSVector|int|float=1):
		location = copy.copy(location_in)
		
		if isinstance(location, str):
			location = dict(ast.literal_eval(location))
			assert(isinstance(location, dict)), "ALSTransform construct from string requires valid dictionary format"

		if not isinstance(location, dict):
			self.location = location
			self.rotation = rotation
			if isinstance(scale, ALSVector):
				self.scale = scale
			else:
				self.scale = ALSVector(scale, scale, scale)
		else:
			assert(scale == 1), "ALSTransform from dict/string cannot take other arguments"
			temporary = ALSTransform.make_from_dict(location)
			self.location = temporary.location
			self.rotation = temporary.rotation
			self.scale = temporary.scale

	def to_dict(self):
		return { "Location": self.location.to_dict(), "Rotation": self.rotation.to_dict(), "Scale": self.scale.to_dict() }

	def __str__(self):
		return "".join(str(self.to_dict()).split())
	
	def make_from_dict(d:dict):
		location = d.get("Location", None)
		rotation = d.get("Rotation", None)
		scale = d.get("Scale", None)
		assert(location != None), "ALSTransform.make_from_dict() requires at least a location"
		if rotation == None:
			rotation = MakeALSVector4()
		if scale == None:
			scale = MakeALSVector3(1,1,1)
		return ALSTransform(ALSVector(location), ALSVector(rotation), ALSVector(scale))

	def make_from_string(transform_string:str):
		return ALSTransform.make_from_dict(dict(ast.literal_eval(transform_string)))

def combine_overrides(*overrides):
	result = str()
	for override in overrides:
		result += override + "<br>"
	return result.removesuffix("<br>")

def sanitize_override(override:str):
	return str().join(override.split()).strip()

def make_function_override(target_value_path:str, function_name:str, *arguments):
	arguments_str = [str(arg) for arg in arguments]
	return sanitize_override(f"{target_value_path};Function;{function_name}({','.join(arguments_str)})")