import sys, ctypes, struct, threading, socket, re, time, logging, atexit, secrets
from ALSLib.ALSOverrideUtility import ALSVector, ALSTransform

try:
    from Queue import Queue
except:
    from queue import Queue  # for Python 3
_L = logging.getLogger(__name__)
_L.handlers = []
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(levelname)s:%(module)s:%(lineno)d:%(message)s"))
_L.addHandler(h)
_L.propagate = False
_L.setLevel(logging.INFO)
fmt = "I"


class SocketMessage(object):
    """
    Define the format of a message. This class is defined similar to the class FNFSMessageHeader in UnrealEngine4, but without CRC check.
    The magic number is from Unreal implementation
    See https://github.com/EpicGames/UnrealEngine/blob/dff3c48be101bb9f84633a733ef79c91c38d9542/Engine/Source/Runtime/Sockets/Public/NetworkMessage.h
    """

    magic = ctypes.c_uint32(0x9E2B83C1).value

    def __init__(self, payload):
        self.magic = SocketMessage.magic
        self.payload_size = ctypes.c_uint32(len(payload)).value

    @classmethod
    def ReceivePayload(cls, socket):

        rbufsize = 0
        rfile = socket.makefile("rb", rbufsize)
        _L.debug("read raw_magic %s", threading.current_thread().name)
        try:
            raw_magic = rfile.read(4)  # socket is disconnected or invalid
        except Exception as e:
            _L.debug("Fail to read raw_magic, %s", e)
            raw_magic = None

        _L.debug(
            "read raw_magic %s done: %s",
            threading.current_thread().name,
            repr(raw_magic),
        )
        if not raw_magic:  # nothing to read
            return None
        magic = struct.unpack(fmt, raw_magic)[0]  # 'I' means unsigned int

        if magic != cls.magic:
            _L.error(
                "Error: receive a malformat message, the message should start from a four bytes uint32 magic number"
            )
            return None
            # The next time it will read four bytes again

        _L.debug("read payload")
        try:
            raw_payload_size = rfile.read(4)
        except Exception as e:
            _L.debug("Fail to read raw_payload_size, %s", e)
            raw_payload_size = None

        if not raw_payload_size:  # nothing to read
            return None
        payload_size = struct.unpack("I", raw_payload_size)[0]
        _L.debug("Receive payload size %d", payload_size)

        # if the message is incomplete, should wait until all the data received
        payload = b""
        remain_size = payload_size
        while remain_size > 0:
            try:
                data = rfile.read(remain_size)
            except Exception as e:
                _L.debug("Fail to read data, %s", e)
                data = None

            if not data:
                return None

            payload += data
            bytes_read = len(
                data
            )  # len(data) is its string length, but we want length of bytes
            assert bytes_read <= remain_size
            remain_size -= bytes_read

        rfile.close()
        return payload

    @classmethod
    def WrapAndSendPayload(cls, socket, payload):

        try:
            wbufsize = -1
            socket_message = SocketMessage(payload)
            wfile = socket.makefile("wb", wbufsize)
            wfile.write(struct.pack(fmt, socket_message.magic))
            wfile.write(struct.pack(fmt, socket_message.payload_size))

            wfile.write(payload)
            wfile.flush()
            wfile.close()  # Close file object, not close the socket
            return True
        except Exception as e:
            _L.error("Fail to send message %s", e)
            return False


class BaseClient(object):

    def __init__(self, endpoint, raw_message_handler):

        self.endpoint = endpoint
        self.raw_message_handler = raw_message_handler
        self.socket = None  # if socket == None, means client is not connected
        self.wait_connected = threading.Event()

        # Start a thread to get data from the socket
        receiving_thread = threading.Thread(target=self.__receiving)
        receiving_thread.daemon = 1
        receiving_thread.start()

    def connect(self, timeout=1):

        if self.isconnected():
            return True

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(self.endpoint)
            self.socket = s
            _L.debug("BaseClient: wait for connection confirm")

            self.wait_connected.clear()
            isset = self.wait_connected.wait(timeout)
            assert isset != None
            if isset:
                return True
            else:
                self.socket = None
                _L.error(
                    "Socket is created, but can not get connection confirm from %s, timeout after %.2f seconds",
                    self.endpoint,
                    timeout,
                )
                return False

        except Exception as e:
            _L.error("Can not connect to %s", str(self.endpoint))
            _L.error("Error %s", e)
            self.socket = None
            return False

    def isconnected(self):
        return self.socket is not None

    def disconnect(self):
        if self.isconnected():
            _L.debug(
                "BaseClient, request disconnect from server in %s",
                threading.current_thread().name,
            )

            self.socket.shutdown(socket.SHUT_WR)
            if self.socket:  # This may also be set to None in the __receiving thread
                self.socket.close()
                self.socket = None
            time.sleep(0.1)

    def __receiving(self):

        _L.debug("BaseClient start receiving in %s", threading.current_thread().name)
        while True:
            if self.isconnected():

                message = SocketMessage.ReceivePayload(self.socket)
                _L.debug("Got server raw message: %s", message)
                if not message:
                    _L.debug("BaseClient: remote disconnected, no more message")
                    self.socket = None
                    continue

                if message.startswith(b"connected"):
                    _L.info("Got connection confirm: %s", repr(message))
                    self.wait_connected.set()
                    continue

                if self.raw_message_handler:
                    self.raw_message_handler(message)  # will block this thread
                else:
                    _L.error("No message handler for raw message %s", message)

    def send(self, message):
        if self.isconnected():
            _L.debug("BaseClient: Send message %s", self.socket)
            SocketMessage.WrapAndSendPayload(self.socket, message)
            return True
        else:
            _L.error("Fail to send message, client is not connected")
            return False


class Client(object):

    def __raw_message_handler(self, raw_message):
        # print 'Waiting for message id %d' % self.message_id
        match = self.raw_message_regexp.match(raw_message)
        if raw_message.startswith(b"Info:"):
            _L.info("this message is not coming from a request")
            match = False

        if match:
            _L.debug("response received, signature match")
            [session_id, message_id, message_body] = (
                match.group(1),
                int(match.group(2)),
                match.group(3),
            )  # TODO: handle multiline response
            # Extract message body from the raw message.
            message_body = raw_message[
                len(match.group(1)) + 1 + len(match.group(2)) + 1 :
            ]
            try:
                message_body = message_body.decode("utf-8")
            except UnicodeDecodeError:
                _L.debug("Exception in decoding the message body")
                pass
            if session_id != self.session_id_as_bytes:
                _L.debug(
                    "Received message not matching session id: expected: %r received: %r",
                    self.session_id,
                    session_id.decode("utf-8"),
                )
                return
            if message_body.startswith("error"):
                print(f"Received error response: {message_body[6:]}")
            if self.print_responses:
                print(f"{message_id}:{message_body}")
            if message_body == "Starting async task":
                self.wait_possible = True
                self.async_IDs.append(message_id)
                return
            elif message_body == "Done!":
                self.async_IDs.remove(message_id)
                if len(self.async_IDs) == 0:
                    self.wait_possible = False
                self.response = message_body
                if not self.waiting_for_async_message:
                    self.wait_response.set()
                return
            if (
                message_id == self.message_id
                and self.waiting_for_async_message is False
            ):
                self.response = message_body
                if not self.waiting_for_async_message:
                    self.wait_response.set()
        else:
            if self.message_handler:

                def do_callback():
                    self.message_handler(raw_message)

                _L.debug("-> adding to the queue for later process")
                self.queue.put(do_callback)
            else:
                # Instead of just dropping this message, give a verbose notice
                _L.error("No message handler to handle message %s", raw_message)

    def __init__(self, endpoint, message_handler=None):
        self.raw_message_regexp = re.compile(b"([a-fA-F0-9]+):(\d{1,8}):(.*)")
        self.session_id = secrets.token_hex(4)
        self.session_id_as_bytes = self.session_id.encode("utf-8")
        self.message_client = BaseClient(endpoint, self.__raw_message_handler)
        self.message_handler = message_handler
        self.message_id = 0
        self.wait_response = threading.Event()
        self.response = ""
        self.waiting_for_async_message = False
        self.wait_possible = False
        self.lock = threading.Lock()
        self.isconnected = self.message_client.isconnected
        self.connect = self.message_client.connect
        self.disconnect = self.message_client.disconnect
        self.async_IDs = []
        self.print_responses = True

        self.queue = Queue()
        self.main_thread = threading.Thread(target=self.worker)
        self.main_thread.daemon = 1
        self.main_thread.start()
        atexit.register(self.disconnect)

    def worker(self):
        while True:
            while self.wait_response.is_set():
                time.sleep(0.2)
            task = self.queue.get()
            task()
            self.queue.task_done()

    def wait_for_task_complete(self):
        if not self.wait_possible:
            print("Wait not possible for the last command")
            return
        self.lock.acquire()
        self.waiting_for_async_message = True
        self.wait_response.clear()
        self.lock.release()
        while self.response != "Done!":
            if "Task took too long " in self.response:
                print("Wait for task timed out")
                break
            if "error" in self.response:
                print("Error running command")
                break
            time.sleep(0.5)
        self.lock.acquire()
        self.waiting_for_async_message = False
        self.wait_possible = False
        self.lock.release()

    def request(self, message, timeout=5):
        if sys.version_info[0] == 3:
            if not isinstance(message, bytes):
                message = message.encode("utf-8")

        def do_request():
            raw_message = b"%s:%d:%s" % (
                self.session_id_as_bytes,
                self.message_id,
                message,
            )
            _L.debug("Request: %s", raw_message.decode("utf-8", errors="replace"))
            if not self.message_client.send(raw_message):
                return None

        # request can only be sent in the main thread, do not support multi-thread submitting request together
        self.lock.acquire()
        if threading.current_thread().name == self.main_thread.name:
            while self.wait_response.is_set():
                time.sleep(0.2)
            do_request()
        else:
            self.queue.put(do_request)
        self.wait_response.clear()
        isset = self.wait_response.wait(timeout)
        self.message_id += 1
        self.wait_response.clear()
        self.lock.release()
        assert isset is not None
        if isset:
            return self.response
        else:
            _L.error(
                "Can not receive a response from server, timeouted after %.2f seconds",
                timeout,
            )
            return None

    def execute(self, message, timeout=5):

        if sys.version_info[0] == 3:
            if not isinstance(message, bytes):
                message = message.encode("utf-8")

        def do_exe():
            raw_message = b"%s:%d:%s" % (
                self.session_id_as_bytes,
                -self.message_id,
                message,
            )
            _L.debug("Execute: %s", raw_message.decode("utf-8", errors="replace"))
            if not self.message_client.send(raw_message):
                return None

        # request can only be sent in the main thread, do not support multi-thread submitting request together
        if threading.current_thread().name == self.main_thread.name:
            do_exe()
        else:
            self.queue.put(do_exe)

        self.wait_response.clear()
        return None

    # Generated API
    def request_help(self, timeout: float = 5):
        """List all available commands."""
        return self.request("Help", timeout)

    def execute_get_simulation_state(self, timeout: float = 5):
        """Returns the current simulation state."""
        self.execute("GetSimulationState", timeout)

    def request_get_simulation_state(self, timeout: float = 5):
        """Returns the current simulation state."""
        return self.request("GetSimulationState", timeout)

    def execute_get_version_info(self, timeout: float = 5):
        """Get the current version and installed plugins."""
        self.execute("GetVersionInfo", timeout)

    def request_get_version_info(self, timeout: float = 5):
        """Get the current version and installed plugins."""
        return self.request("GetVersionInfo", timeout)

    def execute_toggle_pause(self, timeout: float = 5):
        """Pause or continue the Simulation."""
        self.execute("TogglePause", timeout)

    def request_toggle_pause(self, timeout: float = 5):
        """Pause or continue the Simulation."""
        return self.request("TogglePause", timeout)

    def request_open_level(self, levelname: str, timeout: float = 5):
        """Open a level. \n\tUsage: OpenLevel [levelname]"""
        if not (isinstance(levelname, str)):
            print(
                "open_level: levelname has wrong type. Expected str got {}".format(
                    levelname.__class__.__name__
                )
            )
        return_value = self.request("OpenLevel {}".format(levelname), timeout)
        self.wait_for_task_complete()
        return return_value

    def execute_teleport(
        self, posx: float, posy: float, posz: float, timeout: float = 5
    ):
        """Teleport all EgoVehicles to one position. \n\tUsage: Teleport [posx] [posy] [posz]"""
        if not (isinstance(posx, int) or isinstance(posx, float)):
            print(
                "teleport: posx has wrong type. Expected float got {}".format(
                    posx.__class__.__name__
                )
            )
        if not (isinstance(posy, int) or isinstance(posy, float)):
            print(
                "teleport: posy has wrong type. Expected float got {}".format(
                    posy.__class__.__name__
                )
            )
        if not (isinstance(posz, int) or isinstance(posz, float)):
            print(
                "teleport: posz has wrong type. Expected float got {}".format(
                    posz.__class__.__name__
                )
            )
        self.execute("Teleport {} {} {}".format(posx, posy, posz), timeout)

    def request_teleport(
        self, posx: float, posy: float, posz: float, timeout: float = 5
    ):
        """Teleport all EgoVehicles to one position. \n\tUsage: Teleport [posx] [posy] [posz]"""
        if not (isinstance(posx, int) or isinstance(posx, float)):
            print(
                "teleport: posx has wrong type. Expected float got {}".format(
                    posx.__class__.__name__
                )
            )
        if not (isinstance(posy, int) or isinstance(posy, float)):
            print(
                "teleport: posy has wrong type. Expected float got {}".format(
                    posy.__class__.__name__
                )
            )
        if not (isinstance(posz, int) or isinstance(posz, float)):
            print(
                "teleport: posz has wrong type. Expected float got {}".format(
                    posz.__class__.__name__
                )
            )
        return self.request("Teleport {} {} {}".format(posx, posy, posz), timeout)

    def execute_rotate(self, pitch: float, roll: float, yaw: float, timeout: float = 5):
        """Rotate all EgoVehicles. \n\tUsage: [pitch] [roll] [yaw]"""
        if not (isinstance(pitch, int) or isinstance(pitch, float)):
            print(
                "rotate: pitch has wrong type. Expected float got {}".format(
                    pitch.__class__.__name__
                )
            )
        if not (isinstance(roll, int) or isinstance(roll, float)):
            print(
                "rotate: roll has wrong type. Expected float got {}".format(
                    roll.__class__.__name__
                )
            )
        if not (isinstance(yaw, int) or isinstance(yaw, float)):
            print(
                "rotate: yaw has wrong type. Expected float got {}".format(
                    yaw.__class__.__name__
                )
            )
        self.execute("Rotate {} {} {}".format(pitch, roll, yaw), timeout)

    def request_rotate(self, pitch: float, roll: float, yaw: float, timeout: float = 5):
        """Rotate all EgoVehicles. \n\tUsage: [pitch] [roll] [yaw]"""
        if not (isinstance(pitch, int) or isinstance(pitch, float)):
            print(
                "rotate: pitch has wrong type. Expected float got {}".format(
                    pitch.__class__.__name__
                )
            )
        if not (isinstance(roll, int) or isinstance(roll, float)):
            print(
                "rotate: roll has wrong type. Expected float got {}".format(
                    roll.__class__.__name__
                )
            )
        if not (isinstance(yaw, int) or isinstance(yaw, float)):
            print(
                "rotate: yaw has wrong type. Expected float got {}".format(
                    yaw.__class__.__name__
                )
            )
        return self.request("Rotate {} {} {}".format(pitch, roll, yaw), timeout)

    def request_load_scenario(self, scenarioname: str, timeout: float = 5):
        """Load a preexisting Scenario file. \n\tUsage: LoadScenario [scenarioname]"""
        if not (isinstance(scenarioname, str)):
            print(
                "load_scenario: scenarioname has wrong type. Expected str got {}".format(
                    scenarioname.__class__.__name__
                )
            )
        return_value = self.request("LoadScenario {}".format(scenarioname), timeout)
        self.wait_for_task_complete()
        return return_value

    def request_load_scenario_from_string(
        self, scenariostring: str, timeout: float = 5
    ):
        """Load a scenario from a string. \n\tUsage: LoadScenarioFromString [scenariostring]"""
        if not (isinstance(scenariostring, str)):
            print(
                "load_scenario_from_string: scenariostring has wrong type. Expected str got {}".format(
                    scenariostring.__class__.__name__
                )
            )
        return_value = self.request(
            "LoadScenarioFromString {}".format(scenariostring), timeout
        )
        self.wait_for_task_complete()
        return return_value

    def request_load_sub_scene(self, name: str, load: bool, timeout: float = 5):
        """Load or unload one single Scene by name. \n\tUsage: LoadSubScene [name] [true/false]"""
        if not (isinstance(name, str)):
            print(
                "load_sub_scene: name has wrong type. Expected str got {}".format(
                    name.__class__.__name__
                )
            )
        if not (isinstance(load, bool)):
            print(
                "load_sub_scene: load has wrong type. Expected bool got {}".format(
                    load.__class__.__name__
                )
            )
        return_value = self.request("LoadSubScene {} {}".format(name, load), timeout)
        self.wait_for_task_complete()
        return return_value

    def execute_load_situation(self, situationfilename: str, timeout: float = 5):
        """Load a situation. \n\tUsage: LoadSituation [situationfilename]"""
        if not (isinstance(situationfilename, str)):
            print(
                "load_situation: situationfilename has wrong type. Expected str got {}".format(
                    situationfilename.__class__.__name__
                )
            )
        self.execute("LoadSituation {}".format(situationfilename), timeout)

    def request_load_situation(self, situationfilename: str, timeout: float = 5):
        """Load a situation. \n\tUsage: LoadSituation [situationfilename]"""
        if not (isinstance(situationfilename, str)):
            print(
                "load_situation: situationfilename has wrong type. Expected str got {}".format(
                    situationfilename.__class__.__name__
                )
            )
        return self.request("LoadSituation {}".format(situationfilename), timeout)

    def request_load_situation_layer(self, situation: str, timeout: float = 5):
        """Load a situation layer. \n\tUsage: LoadSituationLayer [situation] (individual Scenario file)"""
        if not (isinstance(situation, str)):
            print(
                "load_situation_layer: situation has wrong type. Expected str got {}".format(
                    situation.__class__.__name__
                )
            )
        return_value = self.request("LoadSituationLayer {}".format(situation), timeout)
        self.wait_for_task_complete()
        return return_value

    def execute_destroy_situation(self, timeout: float = 5):
        """Destroys the current situation."""
        self.execute("DestroySituation", timeout)

    def request_destroy_situation(self, timeout: float = 5):
        """Destroys the current situation."""
        return self.request("DestroySituation", timeout)

    def execute_load_weather(self, weatherfilename: str, timeout: float = 5):
        """Load a the weather files if supported by the scene. \n\tUsage: LoadWeather [weatherfilename]"""
        if not (isinstance(weatherfilename, str)):
            print(
                "load_weather: weatherfilename has wrong type. Expected str got {}".format(
                    weatherfilename.__class__.__name__
                )
            )
        self.execute("LoadWeather {}".format(weatherfilename), timeout)

    def request_load_weather(self, weatherfilename: str, timeout: float = 5):
        """Load a the weather files if supported by the scene. \n\tUsage: LoadWeather [weatherfilename]"""
        if not (isinstance(weatherfilename, str)):
            print(
                "load_weather: weatherfilename has wrong type. Expected str got {}".format(
                    weatherfilename.__class__.__name__
                )
            )
        return self.request("LoadWeather {}".format(weatherfilename), timeout)

    def execute_load_sea_state(self, seastatefilename: str, timeout: float = 5):
        """Load one of the sea state files if supported by the scene. \n\tUsage: LoadSeaState [seastatefilename]"""
        if not (isinstance(seastatefilename, str)):
            print(
                "load_sea_state: seastatefilename has wrong type. Expected str got {}".format(
                    seastatefilename.__class__.__name__
                )
            )
        self.execute("LoadSeaState {}".format(seastatefilename), timeout)

    def request_load_sea_state(self, seastatefilename: str, timeout: float = 5):
        """Load one of the sea state files if supported by the scene. \n\tUsage: LoadSeaState [seastatefilename]"""
        if not (isinstance(seastatefilename, str)):
            print(
                "load_sea_state: seastatefilename has wrong type. Expected str got {}".format(
                    seastatefilename.__class__.__name__
                )
            )
        return self.request("LoadSeaState {}".format(seastatefilename), timeout)

    def execute_cmd_cycle_pawns(self, timeout: float = 5):
        """Possess the next pawn. \n\tUsage: CmdCyclePawns"""
        self.execute("CmdCyclePawns", timeout)

    def request_cmd_cycle_pawns(self, timeout: float = 5):
        """Possess the next pawn. \n\tUsage: CmdCyclePawns"""
        return self.request("CmdCyclePawns", timeout)

    def execute_cmd_cycle_egos(self, timeout: float = 5):
        """Possess the next Ego or detach if last one. \n\tUsage: CmdCycleEgos"""
        self.execute("CmdCycleEgos", timeout)

    def request_cmd_cycle_egos(self, timeout: float = 5):
        """Possess the next Ego or detach if last one. \n\tUsage: CmdCycleEgos"""
        return self.request("CmdCycleEgos", timeout)

    def execute_cmd_cycle_cameras(self, timeout: float = 5):
        """Possess the next Camera. \n\tUsage: CmdCycleCameras"""
        self.execute("CmdCycleCameras", timeout)

    def request_cmd_cycle_cameras(self, timeout: float = 5):
        """Possess the next Camera. \n\tUsage: CmdCycleCameras"""
        return self.request("CmdCycleCameras", timeout)

    def execute_set_parameter_int(self, name: str, value: int, timeout: float = 5):
        """Sets the value of the given variable. \n\tUsage: SetParameterInt [name] [value]"""
        if not (isinstance(name, str)):
            print(
                "set_parameter_int: name has wrong type. Expected str got {}".format(
                    name.__class__.__name__
                )
            )
        if not (isinstance(value, int)):
            print(
                "set_parameter_int: value has wrong type. Expected int got {}".format(
                    value.__class__.__name__
                )
            )
        self.execute("SetParameterInt {} {}".format(name, value), timeout)

    def request_set_parameter_int(self, name: str, value: int, timeout: float = 5):
        """Sets the value of the given variable. \n\tUsage: SetParameterInt [name] [value]"""
        if not (isinstance(name, str)):
            print(
                "set_parameter_int: name has wrong type. Expected str got {}".format(
                    name.__class__.__name__
                )
            )
        if not (isinstance(value, int)):
            print(
                "set_parameter_int: value has wrong type. Expected int got {}".format(
                    value.__class__.__name__
                )
            )
        return self.request("SetParameterInt {} {}".format(name, value), timeout)

    def execute_set_parameter_float(self, name: str, value: float, timeout: float = 5):
        """Sets the value of the given variable. \n\tUsage: SetParameterFloat [name] [value]"""
        if not (isinstance(name, str)):
            print(
                "set_parameter_float: name has wrong type. Expected str got {}".format(
                    name.__class__.__name__
                )
            )
        if not (isinstance(value, int) or isinstance(value, float)):
            print(
                "set_parameter_float: value has wrong type. Expected float got {}".format(
                    value.__class__.__name__
                )
            )
        self.execute("SetParameterFloat {} {}".format(name, value), timeout)

    def request_set_parameter_float(self, name: str, value: float, timeout: float = 5):
        """Sets the value of the given variable. \n\tUsage: SetParameterFloat [name] [value]"""
        if not (isinstance(name, str)):
            print(
                "set_parameter_float: name has wrong type. Expected str got {}".format(
                    name.__class__.__name__
                )
            )
        if not (isinstance(value, int) or isinstance(value, float)):
            print(
                "set_parameter_float: value has wrong type. Expected float got {}".format(
                    value.__class__.__name__
                )
            )
        return self.request("SetParameterFloat {} {}".format(name, value), timeout)

    def execute_set_simulation_speed_factor(self, value: float, timeout: float = 5):
        """Sets speed of the simulation. (1.0f is the default) \n\tUsage: SetSimulationSpeedFactor [value]"""
        if not (isinstance(value, int) or isinstance(value, float)):
            print(
                "set_simulation_speed_factor: value has wrong type. Expected float got {}".format(
                    value.__class__.__name__
                )
            )
        self.execute("SetSimulationSpeedFactor {}".format(value), timeout)

    def request_set_simulation_speed_factor(self, value: float, timeout: float = 5):
        """Sets speed of the simulation. (1.0f is the default) \n\tUsage: SetSimulationSpeedFactor [value]"""
        if not (isinstance(value, int) or isinstance(value, float)):
            print(
                "set_simulation_speed_factor: value has wrong type. Expected float got {}".format(
                    value.__class__.__name__
                )
            )
        return self.request("SetSimulationSpeedFactor {}".format(value), timeout)

    def execute_set_simulation_speed_to_max(self, timeout: float = 5):
        """Set the simulation speed to as fast as possible."""
        self.execute("SetSimulationSpeedToMax", timeout)

    def request_set_simulation_speed_to_max(self, timeout: float = 5):
        """Set the simulation speed to as fast as possible."""
        return self.request("SetSimulationSpeedToMax", timeout)

    def execute_set_simulation_speed_to_real_time(self, timeout: float = 5):
        """Sets the simulation speed to real time. (sets speed to 1.0f)"""
        self.execute("SetSimulationSpeedToRealTime", timeout)

    def request_set_simulation_speed_to_real_time(self, timeout: float = 5):
        """Sets the simulation speed to real time. (sets speed to 1.0f)"""
        return self.request("SetSimulationSpeedToRealTime", timeout)

    def execute_reload_all_cameras(self, timeout: float = 5):
        """Reload advanced settings for each camera."""
        self.execute("ReloadAllCameras", timeout)

    def request_reload_all_cameras(self, timeout: float = 5):
        """Reload advanced settings for each camera."""
        return self.request("ReloadAllCameras", timeout)

    def execute_get_global_seed(self, timeout: float = 5):
        """Returns the current global seed."""
        self.execute("GetGlobalSeed", timeout)

    def request_get_global_seed(self, timeout: float = 5):
        """Returns the current global seed."""
        return self.request("GetGlobalSeed", timeout)

    def execute_start_seed_override(self, seed: int, timeout: float = 5):
        """Starts overriding the global seed. \n\tUsage: StartSeedOverride [seed]"""
        if not (isinstance(seed, int)):
            print(
                "start_seed_override: seed has wrong type. Expected int got {}".format(
                    seed.__class__.__name__
                )
            )
        self.execute("StartSeedOverride {}".format(seed), timeout)

    def request_start_seed_override(self, seed: int, timeout: float = 5):
        """Starts overriding the global seed. \n\tUsage: StartSeedOverride [seed]"""
        if not (isinstance(seed, int)):
            print(
                "start_seed_override: seed has wrong type. Expected int got {}".format(
                    seed.__class__.__name__
                )
            )
        return self.request("StartSeedOverride {}".format(seed), timeout)

    def execute_stop_seed_override(self, timeout: float = 5):
        """Stop overriding the global seed."""
        self.execute("StopSeedOverride", timeout)

    def request_stop_seed_override(self, timeout: float = 5):
        """Stop overriding the global seed."""
        return self.request("StopSeedOverride", timeout)

    def execute_spawn_debug_sphere(
        self,
        posx: float,
        posy: float,
        posz: float,
        scalex: float,
        scaley: float,
        scalez: float,
        sphere_color: str,
        duration_in_seconds: float,
        timeout: float = 5,
    ):
        """Spawns a debugging sphere. \n\tUsage: SpawnDebugSphere [posx] [posy] [posz] [scalex] [scaley] [scalez] [redphere/greenphere/bluephere] [duration in seconds]"""
        if not (isinstance(posx, int) or isinstance(posx, float)):
            print(
                "spawn_debug_sphere: posx has wrong type. Expected float got {}".format(
                    posx.__class__.__name__
                )
            )
        if not (isinstance(posy, int) or isinstance(posy, float)):
            print(
                "spawn_debug_sphere: posy has wrong type. Expected float got {}".format(
                    posy.__class__.__name__
                )
            )
        if not (isinstance(posz, int) or isinstance(posz, float)):
            print(
                "spawn_debug_sphere: posz has wrong type. Expected float got {}".format(
                    posz.__class__.__name__
                )
            )
        if not (isinstance(scalex, int) or isinstance(scalex, float)):
            print(
                "spawn_debug_sphere: scalex has wrong type. Expected float got {}".format(
                    scalex.__class__.__name__
                )
            )
        if not (isinstance(scaley, int) or isinstance(scaley, float)):
            print(
                "spawn_debug_sphere: scaley has wrong type. Expected float got {}".format(
                    scaley.__class__.__name__
                )
            )
        if not (isinstance(scalez, int) or isinstance(scalez, float)):
            print(
                "spawn_debug_sphere: scalez has wrong type. Expected float got {}".format(
                    scalez.__class__.__name__
                )
            )
        if not (isinstance(sphere_color, str)):
            print(
                "spawn_debug_sphere: sphere_color has wrong type. Expected str got {}".format(
                    sphere_color.__class__.__name__
                )
            )
        if not (
            isinstance(duration_in_seconds, int)
            or isinstance(duration_in_seconds, float)
        ):
            print(
                "spawn_debug_sphere: duration_in_seconds has wrong type. Expected float got {}".format(
                    duration_in_seconds.__class__.__name__
                )
            )
        self.execute(
            "SpawnDebugSphere {} {} {} {} {} {} {} {}".format(
                posx,
                posy,
                posz,
                scalex,
                scaley,
                scalez,
                sphere_color,
                duration_in_seconds,
            ),
            timeout,
        )

    def request_spawn_debug_sphere(
        self,
        posx: float,
        posy: float,
        posz: float,
        scalex: float,
        scaley: float,
        scalez: float,
        sphere_color: str,
        duration_in_seconds: float,
        timeout: float = 5,
    ):
        """Spawns a debugging sphere. \n\tUsage: SpawnDebugSphere [posx] [posy] [posz] [scalex] [scaley] [scalez] [redphere/greenphere/bluephere] [duration in seconds]"""
        if not (isinstance(posx, int) or isinstance(posx, float)):
            print(
                "spawn_debug_sphere: posx has wrong type. Expected float got {}".format(
                    posx.__class__.__name__
                )
            )
        if not (isinstance(posy, int) or isinstance(posy, float)):
            print(
                "spawn_debug_sphere: posy has wrong type. Expected float got {}".format(
                    posy.__class__.__name__
                )
            )
        if not (isinstance(posz, int) or isinstance(posz, float)):
            print(
                "spawn_debug_sphere: posz has wrong type. Expected float got {}".format(
                    posz.__class__.__name__
                )
            )
        if not (isinstance(scalex, int) or isinstance(scalex, float)):
            print(
                "spawn_debug_sphere: scalex has wrong type. Expected float got {}".format(
                    scalex.__class__.__name__
                )
            )
        if not (isinstance(scaley, int) or isinstance(scaley, float)):
            print(
                "spawn_debug_sphere: scaley has wrong type. Expected float got {}".format(
                    scaley.__class__.__name__
                )
            )
        if not (isinstance(scalez, int) or isinstance(scalez, float)):
            print(
                "spawn_debug_sphere: scalez has wrong type. Expected float got {}".format(
                    scalez.__class__.__name__
                )
            )
        if not (isinstance(sphere_color, str)):
            print(
                "spawn_debug_sphere: sphere_color has wrong type. Expected str got {}".format(
                    sphere_color.__class__.__name__
                )
            )
        if not (
            isinstance(duration_in_seconds, int)
            or isinstance(duration_in_seconds, float)
        ):
            print(
                "spawn_debug_sphere: duration_in_seconds has wrong type. Expected float got {}".format(
                    duration_in_seconds.__class__.__name__
                )
            )
        return self.request(
            "SpawnDebugSphere {} {} {} {} {} {} {} {}".format(
                posx,
                posy,
                posz,
                scalex,
                scaley,
                scalez,
                sphere_color,
                duration_in_seconds,
            ),
            timeout,
        )

    def execute_remove_debug_objects(self, timeout: float = 5):
        """Remove all debug objects from the level."""
        self.execute("RemoveDebugObjects", timeout)

    def request_remove_debug_objects(self, timeout: float = 5):
        """Remove all debug objects from the level."""
        return self.request("RemoveDebugObjects", timeout)

    def execute_set_cloud_style(self, style: str, timeout: float = 5):
        """Change cloud style. \n\tUsage: SetCloudStyle [Clear/Cloudy/Overcast/StormClouds]"""
        if not (isinstance(style, str)):
            print(
                "set_cloud_style: style has wrong type. Expected str got {}".format(
                    style.__class__.__name__
                )
            )
        self.execute("SetCloudStyle {}".format(style), timeout)

    def request_set_cloud_style(self, style: str, timeout: float = 5):
        """Change cloud style. \n\tUsage: SetCloudStyle [Clear/Cloudy/Overcast/StormClouds]"""
        if not (isinstance(style, str)):
            print(
                "set_cloud_style: style has wrong type. Expected str got {}".format(
                    style.__class__.__name__
                )
            )
        return self.request("SetCloudStyle {}".format(style), timeout)

    def execute_set_time_of_day(self, time_ISO_8601_string: str, timeout: float = 5):
        """Change time of day. \n\tUsage: SetTimeOfDay [time ISO 8601 string]"""
        if not (isinstance(time_ISO_8601_string, str)):
            print(
                "set_time_of_day: time_ISO_8601_string has wrong type. Expected str got {}".format(
                    time_ISO_8601_string.__class__.__name__
                )
            )
        self.execute("SetTimeOfDay {}".format(time_ISO_8601_string), timeout)

    def request_set_time_of_day(self, time_ISO_8601_string: str, timeout: float = 5):
        """Change time of day. \n\tUsage: SetTimeOfDay [time ISO 8601 string]"""
        if not (isinstance(time_ISO_8601_string, str)):
            print(
                "set_time_of_day: time_ISO_8601_string has wrong type. Expected str got {}".format(
                    time_ISO_8601_string.__class__.__name__
                )
            )
        return self.request("SetTimeOfDay {}".format(time_ISO_8601_string), timeout)

    def execute_set_cloud_coverage(self, cloud_style: float, timeout: float = 5):
        """Change cloud coverage. \n\tUsage: SetCloudCoverage [cloud density]"""
        if not (isinstance(cloud_style, int) or isinstance(cloud_style, float)):
            print(
                "set_cloud_coverage: cloud_style has wrong type. Expected float got {}".format(
                    cloud_style.__class__.__name__
                )
            )
        self.execute("SetCloudCoverage {}".format(cloud_style), timeout)

    def request_set_cloud_coverage(self, cloud_style: float, timeout: float = 5):
        """Change cloud coverage. \n\tUsage: SetCloudCoverage [cloud density]"""
        if not (isinstance(cloud_style, int) or isinstance(cloud_style, float)):
            print(
                "set_cloud_coverage: cloud_style has wrong type. Expected float got {}".format(
                    cloud_style.__class__.__name__
                )
            )
        return self.request("SetCloudCoverage {}".format(cloud_style), timeout)

    def execute_randomize_weather(self, timeout: float = 5):
        """Randomize weather based on loaded random variability."""
        self.execute("RandomizeWeather", timeout)

    def request_randomize_weather(self, timeout: float = 5):
        """Randomize weather based on loaded random variability."""
        return self.request("RandomizeWeather", timeout)

    def execute_set_season(self, season: str, timeout: float = 5):
        """Set season. \n\tUsage: SetSeason [winter/spring/summer/fall]"""
        if not (isinstance(season, str)):
            print(
                "set_season: season has wrong type. Expected str got {}".format(
                    season.__class__.__name__
                )
            )
        self.execute("SetSeason {}".format(season), timeout)

    def request_set_season(self, season: str, timeout: float = 5):
        """Set season. \n\tUsage: SetSeason [winter/spring/summer/fall]"""
        if not (isinstance(season, str)):
            print(
                "set_season: season has wrong type. Expected str got {}".format(
                    season.__class__.__name__
                )
            )
        return self.request("SetSeason {}".format(season), timeout)

    def execute_set_ground_wetness_type(self, type: str, timeout: float = 5):
        """Set ground wetness type. \n\tUsage: SetGroundWetnessType [wet/snowy/dry]"""
        if not (isinstance(type, str)):
            print(
                "set_ground_wetness_type: type has wrong type. Expected str got {}".format(
                    type.__class__.__name__
                )
            )
        self.execute("SetGroundWetnessType {}".format(type), timeout)

    def request_set_ground_wetness_type(self, type: str, timeout: float = 5):
        """Set ground wetness type. \n\tUsage: SetGroundWetnessType [wet/snowy/dry]"""
        if not (isinstance(type, str)):
            print(
                "set_ground_wetness_type: type has wrong type. Expected str got {}".format(
                    type.__class__.__name__
                )
            )
        return self.request("SetGroundWetnessType {}".format(type), timeout)

    def execute_set_ground_wetness_percent(self, percent: float, timeout: float = 5):
        """Set ground wetness percent. \n\tUsage: SetGroundWetnessPercent [percent]"""
        if not (isinstance(percent, int) or isinstance(percent, float)):
            print(
                "set_ground_wetness_percent: percent has wrong type. Expected float got {}".format(
                    percent.__class__.__name__
                )
            )
        self.execute("SetGroundWetnessPercent {}".format(percent), timeout)

    def request_set_ground_wetness_percent(self, percent: float, timeout: float = 5):
        """Set ground wetness percent. \n\tUsage: SetGroundWetnessPercent [percent]"""
        if not (isinstance(percent, int) or isinstance(percent, float)):
            print(
                "set_ground_wetness_percent: percent has wrong type. Expected float got {}".format(
                    percent.__class__.__name__
                )
            )
        return self.request("SetGroundWetnessPercent {}".format(percent), timeout)

    def execute_set_rain_type(self, type: str, timeout: float = 5):
        """Set rain type. \n\tUsage: SetRainType [normal/snow/hail/leaves/norain]"""
        if not (isinstance(type, str)):
            print(
                "set_rain_type: type has wrong type. Expected str got {}".format(
                    type.__class__.__name__
                )
            )
        self.execute("SetRainType {}".format(type), timeout)

    def request_set_rain_type(self, type: str, timeout: float = 5):
        """Set rain type. \n\tUsage: SetRainType [normal/snow/hail/leaves/norain]"""
        if not (isinstance(type, str)):
            print(
                "set_rain_type: type has wrong type. Expected str got {}".format(
                    type.__class__.__name__
                )
            )
        return self.request("SetRainType {}".format(type), timeout)

    def execute_set_rain_intensity(self, intensity: float, timeout: float = 5):
        """Set rain intensity. \n\tUsage: SetRainIntensity [intensity]"""
        if not (isinstance(intensity, int) or isinstance(intensity, float)):
            print(
                "set_rain_intensity: intensity has wrong type. Expected float got {}".format(
                    intensity.__class__.__name__
                )
            )
        self.execute("SetRainIntensity {}".format(intensity), timeout)

    def request_set_rain_intensity(self, intensity: float, timeout: float = 5):
        """Set rain intensity. \n\tUsage: SetRainIntensity [intensity]"""
        if not (isinstance(intensity, int) or isinstance(intensity, float)):
            print(
                "set_rain_intensity: intensity has wrong type. Expected float got {}".format(
                    intensity.__class__.__name__
                )
            )
        return self.request("SetRainIntensity {}".format(intensity), timeout)

    def execute_get_weather_status(self, timeout: float = 5):
        """Get the weather.ini file corresponding to the current weather values."""
        self.execute("GetWeatherStatus", timeout)

    def request_get_weather_status(self, timeout: float = 5):
        """Get the weather.ini file corresponding to the current weather values."""
        return self.request("GetWeatherStatus", timeout)

    def execute_get_sea_state(self, timeout: float = 5):
        """Get the seastate.ini file corresponding to the current seastate values."""
        self.execute("GetSeaState", timeout)

    def request_get_sea_state(self, timeout: float = 5):
        """Get the seastate.ini file corresponding to the current seastate values."""
        return self.request("GetSeaState", timeout)

    def execute_get_scenario(self, timeout: float = 5):
        """Get the Scenario information for the current scenario."""
        self.execute("GetScenario", timeout)

    def request_get_scenario(self, timeout: float = 5):
        """Get the Scenario information for the current scenario."""
        return self.request("GetScenario", timeout)

    def execute_push_scenario(self, timeout: float = 5):
        """Push the Scenario. (Currently does nothing)"""
        self.execute("PushScenario", timeout)

    def request_push_scenario(self, timeout: float = 5):
        """Push the Scenario. (Currently does nothing)"""
        return self.request("PushScenario", timeout)

    def execute_replay(self, filename: str, timeout: float = 5):
        """Replay from file. \n\tUsage: Replay [filename]"""
        if not (isinstance(filename, str)):
            print(
                "replay: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        self.execute("Replay {}".format(filename), timeout)

    def request_replay(self, filename: str, timeout: float = 5):
        """Replay from file. \n\tUsage: Replay [filename]"""
        if not (isinstance(filename, str)):
            print(
                "replay: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        return self.request("Replay {}".format(filename), timeout)

    def execute_load_situation_with_overrides(
        self, situation: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the situation files \n\tUsage: LoadSituationWithOverrides [situation] [overridefile/overridestring]"""
        if not (isinstance(situation, str)):
            print(
                "load_situation_with_overrides: situation has wrong type. Expected str got {}".format(
                    situation.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_situation_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        self.execute(
            "LoadSituationWithOverrides {} {}".format(situation, override), timeout
        )

    def request_load_situation_with_overrides(
        self, situation: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the situation files \n\tUsage: LoadSituationWithOverrides [situation] [overridefile/overridestring]"""
        if not (isinstance(situation, str)):
            print(
                "load_situation_with_overrides: situation has wrong type. Expected str got {}".format(
                    situation.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_situation_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        return self.request(
            "LoadSituationWithOverrides {} {}".format(situation, override), timeout
        )

    def request_load_situation_layer_with_overrides(
        self, situation: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the situation layer files \n\tUsage: LoadSituationLayerWithOverrides [situation] [overridefile/overridestring]"""
        if not (isinstance(situation, str)):
            print(
                "load_situation_layer_with_overrides: situation has wrong type. Expected str got {}".format(
                    situation.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_situation_layer_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        return_value = self.request(
            "LoadSituationLayerWithOverrides {} {}".format(situation, override), timeout
        )
        self.wait_for_task_complete()
        return return_value

    def execute_load_weather_with_overrides(
        self, weather_file_name: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the weather files if supported by the scene. \n\tUsage: LoadWeatherWithOverrides [weather file name] [path to override file/override file content string]"""
        if not (isinstance(weather_file_name, str)):
            print(
                "load_weather_with_overrides: weather_file_name has wrong type. Expected str got {}".format(
                    weather_file_name.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_weather_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        self.execute(
            "LoadWeatherWithOverrides {} {}".format(weather_file_name, override),
            timeout,
        )

    def request_load_weather_with_overrides(
        self, weather_file_name: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the weather files if supported by the scene. \n\tUsage: LoadWeatherWithOverrides [weather file name] [path to override file/override file content string]"""
        if not (isinstance(weather_file_name, str)):
            print(
                "load_weather_with_overrides: weather_file_name has wrong type. Expected str got {}".format(
                    weather_file_name.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_weather_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        return self.request(
            "LoadWeatherWithOverrides {} {}".format(weather_file_name, override),
            timeout,
        )

    def execute_load_sea_state_with_overrides(
        self, sea_state_file_name: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the sea state files if supported by the scene. \n\tUsage: LoadSeaStateWithOverrides [sea state file name] [path to override file/override file content string]"""
        if not (isinstance(sea_state_file_name, str)):
            print(
                "load_sea_state_with_overrides: sea_state_file_name has wrong type. Expected str got {}".format(
                    sea_state_file_name.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_sea_state_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        self.execute(
            "LoadSeaStateWithOverrides {} {}".format(sea_state_file_name, override),
            timeout,
        )

    def request_load_sea_state_with_overrides(
        self, sea_state_file_name: str, override: str, timeout: float = 5
    ):
        """Load and override variables for one of the sea state files if supported by the scene. \n\tUsage: LoadSeaStateWithOverrides [sea state file name] [path to override file/override file content string]"""
        if not (isinstance(sea_state_file_name, str)):
            print(
                "load_sea_state_with_overrides: sea_state_file_name has wrong type. Expected str got {}".format(
                    sea_state_file_name.__class__.__name__
                )
            )
        if not (isinstance(override, str)):
            print(
                "load_sea_state_with_overrides: override has wrong type. Expected str got {}".format(
                    override.__class__.__name__
                )
            )
        return self.request(
            "LoadSeaStateWithOverrides {} {}".format(sea_state_file_name, override),
            timeout,
        )

    def request_load_scenario_with_overrides(
        self, scenarioname: str, situation_overridefile: str, timeout: float = 5
    ):
        """Load and override variables for a scenario. \n\tUsage: LoadScenarioWithOverrides [scenarioname] [situationoverridefile]"""
        if not (isinstance(scenarioname, str)):
            print(
                "load_scenario_with_overrides: scenarioname has wrong type. Expected str got {}".format(
                    scenarioname.__class__.__name__
                )
            )
        if not (isinstance(situation_overridefile, str)):
            print(
                "load_scenario_with_overrides: situation_overridefile has wrong type. Expected str got {}".format(
                    situation_overridefile.__class__.__name__
                )
            )
        return_value = self.request(
            "LoadScenarioWithOverrides {} {}".format(
                scenarioname, situation_overridefile
            ),
            timeout,
        )
        self.wait_for_task_complete()
        return return_value

    def execute_get_latest_replay_files(self, timeout: float = 5):
        """Get the 2 latest replay files."""
        self.execute("GetLatestReplayFiles", timeout)

    def request_get_latest_replay_files(self, timeout: float = 5):
        """Get the 2 latest replay files."""
        return self.request("GetLatestReplayFiles", timeout)

    def execute_teleport_object(
        self,
        object: str,
        newx: float,
        newy: float,
        newz: float,
        filename: str,
        timeout: float = 5,
    ):
        """Teleport specified object to specified location. \n\tUsage: TeleportObject [path/alias/objectname] [newx] [newy] [newz] [filename]"""
        if not (isinstance(object, str)):
            print(
                "teleport_object: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(newx, int) or isinstance(newx, float)):
            print(
                "teleport_object: newx has wrong type. Expected float got {}".format(
                    newx.__class__.__name__
                )
            )
        if not (isinstance(newy, int) or isinstance(newy, float)):
            print(
                "teleport_object: newy has wrong type. Expected float got {}".format(
                    newy.__class__.__name__
                )
            )
        if not (isinstance(newz, int) or isinstance(newz, float)):
            print(
                "teleport_object: newz has wrong type. Expected float got {}".format(
                    newz.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "teleport_object: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        self.execute(
            "TeleportObject {} {} {} {} {}".format(object, newx, newy, newz, filename),
            timeout,
        )

    def request_teleport_object(
        self,
        object: str,
        newx: float,
        newy: float,
        newz: float,
        filename: str,
        timeout: float = 5,
    ):
        """Teleport specified object to specified location. \n\tUsage: TeleportObject [path/alias/objectname] [newx] [newy] [newz] [filename]"""
        if not (isinstance(object, str)):
            print(
                "teleport_object: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(newx, int) or isinstance(newx, float)):
            print(
                "teleport_object: newx has wrong type. Expected float got {}".format(
                    newx.__class__.__name__
                )
            )
        if not (isinstance(newy, int) or isinstance(newy, float)):
            print(
                "teleport_object: newy has wrong type. Expected float got {}".format(
                    newy.__class__.__name__
                )
            )
        if not (isinstance(newz, int) or isinstance(newz, float)):
            print(
                "teleport_object: newz has wrong type. Expected float got {}".format(
                    newz.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "teleport_object: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        return self.request(
            "TeleportObject {} {} {} {} {}".format(object, newx, newy, newz, filename),
            timeout,
        )

    def execute_teleport_object_gps(
        self,
        object: str,
        newlatitude: float,
        newlongitude: float,
        filename: str,
        timeout: float = 5,
    ):
        """Teleport specified object to specified GPS coordinates. \n\tUsage: TeleportObjectGPS [path/alias/objectname] [newlatitude] [newlongitude] [filename]"""
        if not (isinstance(object, str)):
            print(
                "teleport_object_gps: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(newlatitude, int) or isinstance(newlatitude, float)):
            print(
                "teleport_object_gps: newlatitude has wrong type. Expected float got {}".format(
                    newlatitude.__class__.__name__
                )
            )
        if not (isinstance(newlongitude, int) or isinstance(newlongitude, float)):
            print(
                "teleport_object_gps: newlongitude has wrong type. Expected float got {}".format(
                    newlongitude.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "teleport_object_gps: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        self.execute(
            "TeleportObjectGPS {} {} {} {}".format(
                object, newlatitude, newlongitude, filename
            ),
            timeout,
        )

    def request_teleport_object_gps(
        self,
        object: str,
        newlatitude: float,
        newlongitude: float,
        filename: str,
        timeout: float = 5,
    ):
        """Teleport specified object to specified GPS coordinates. \n\tUsage: TeleportObjectGPS [path/alias/objectname] [newlatitude] [newlongitude] [filename]"""
        if not (isinstance(object, str)):
            print(
                "teleport_object_gps: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(newlatitude, int) or isinstance(newlatitude, float)):
            print(
                "teleport_object_gps: newlatitude has wrong type. Expected float got {}".format(
                    newlatitude.__class__.__name__
                )
            )
        if not (isinstance(newlongitude, int) or isinstance(newlongitude, float)):
            print(
                "teleport_object_gps: newlongitude has wrong type. Expected float got {}".format(
                    newlongitude.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "teleport_object_gps: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        return self.request(
            "TeleportObjectGPS {} {} {} {}".format(
                object, newlatitude, newlongitude, filename
            ),
            timeout,
        )

    def execute_rotate_object(
        self,
        object: str,
        newpitch: float,
        newyaw: float,
        newroll: float,
        filename: str,
        timeout: float = 5,
    ):
        """Rotate specified object to specified rotation. \n\tUsage: RotateObject [path/alias/objectname] [newpitch] [newyaw] [newroll] [filename]"""
        if not (isinstance(object, str)):
            print(
                "rotate_object: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(newpitch, int) or isinstance(newpitch, float)):
            print(
                "rotate_object: newpitch has wrong type. Expected float got {}".format(
                    newpitch.__class__.__name__
                )
            )
        if not (isinstance(newyaw, int) or isinstance(newyaw, float)):
            print(
                "rotate_object: newyaw has wrong type. Expected float got {}".format(
                    newyaw.__class__.__name__
                )
            )
        if not (isinstance(newroll, int) or isinstance(newroll, float)):
            print(
                "rotate_object: newroll has wrong type. Expected float got {}".format(
                    newroll.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "rotate_object: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        self.execute(
            "RotateObject {} {} {} {} {}".format(
                object, newpitch, newyaw, newroll, filename
            ),
            timeout,
        )

    def request_rotate_object(
        self,
        object: str,
        newpitch: float,
        newyaw: float,
        newroll: float,
        filename: str,
        timeout: float = 5,
    ):
        """Rotate specified object to specified rotation. \n\tUsage: RotateObject [path/alias/objectname] [newpitch] [newyaw] [newroll] [filename]"""
        if not (isinstance(object, str)):
            print(
                "rotate_object: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(newpitch, int) or isinstance(newpitch, float)):
            print(
                "rotate_object: newpitch has wrong type. Expected float got {}".format(
                    newpitch.__class__.__name__
                )
            )
        if not (isinstance(newyaw, int) or isinstance(newyaw, float)):
            print(
                "rotate_object: newyaw has wrong type. Expected float got {}".format(
                    newyaw.__class__.__name__
                )
            )
        if not (isinstance(newroll, int) or isinstance(newroll, float)):
            print(
                "rotate_object: newroll has wrong type. Expected float got {}".format(
                    newroll.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "rotate_object: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        return self.request(
            "RotateObject {} {} {} {} {}".format(
                object, newpitch, newyaw, newroll, filename
            ),
            timeout,
        )

    def execute_set_physics_active(self, object: str, active: bool, timeout: float = 5):
        """Set physics active for specified object. \n\tUsage: SetPhysicsActive [path/alias/objectname] [true/false]"""
        if not (isinstance(object, str)):
            print(
                "set_physics_active: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(active, bool)):
            print(
                "set_physics_active: active has wrong type. Expected bool got {}".format(
                    active.__class__.__name__
                )
            )
        self.execute("SetPhysicsActive {} {}".format(object, active), timeout)

    def request_set_physics_active(self, object: str, active: bool, timeout: float = 5):
        """Set physics active for specified object. \n\tUsage: SetPhysicsActive [path/alias/objectname] [true/false]"""
        if not (isinstance(object, str)):
            print(
                "set_physics_active: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(active, bool)):
            print(
                "set_physics_active: active has wrong type. Expected bool got {}".format(
                    active.__class__.__name__
                )
            )
        return self.request("SetPhysicsActive {} {}".format(object, active), timeout)

    def execute_set_object_visibility(
        self, object: str, visible: bool, timeout: float = 5
    ):
        """Set object visibility. \n\tUsage: SetObjectVisibility [path/alias/objectname] [true/false]"""
        if not (isinstance(object, str)):
            print(
                "set_object_visibility: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(visible, bool)):
            print(
                "set_object_visibility: visible has wrong type. Expected bool got {}".format(
                    visible.__class__.__name__
                )
            )
        self.execute("SetObjectVisibility {} {}".format(object, visible), timeout)

    def request_set_object_visibility(
        self, object: str, visible: bool, timeout: float = 5
    ):
        """Set object visibility. \n\tUsage: SetObjectVisibility [path/alias/objectname] [true/false]"""
        if not (isinstance(object, str)):
            print(
                "set_object_visibility: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(visible, bool)):
            print(
                "set_object_visibility: visible has wrong type. Expected bool got {}".format(
                    visible.__class__.__name__
                )
            )
        return self.request(
            "SetObjectVisibility {} {}".format(object, visible), timeout
        )

    def execute_set_object_property(
        self, object: str, variable: str, value: str, timeout: float = 5
    ):
        """Set object property. \n\tUsage: SetObjectProperty [path/alias/objectname] [variable] [value]"""
        if not (isinstance(object, str)):
            print(
                "set_object_property: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(variable, str)):
            print(
                "set_object_property: variable has wrong type. Expected str got {}".format(
                    variable.__class__.__name__
                )
            )
        if not (isinstance(value, str)):
            print(
                "set_object_property: value has wrong type. Expected str got {}".format(
                    value.__class__.__name__
                )
            )
        self.execute(
            "SetObjectProperty {} {} {}".format(object, variable, value), timeout
        )

    def request_set_object_property(
        self, object: str, variable: str, value: str, timeout: float = 5
    ):
        """Set object property. \n\tUsage: SetObjectProperty [path/alias/objectname] [variable] [value]"""
        if not (isinstance(object, str)):
            print(
                "set_object_property: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(variable, str)):
            print(
                "set_object_property: variable has wrong type. Expected str got {}".format(
                    variable.__class__.__name__
                )
            )
        if not (isinstance(value, str)):
            print(
                "set_object_property: value has wrong type. Expected str got {}".format(
                    value.__class__.__name__
                )
            )
        return self.request(
            "SetObjectProperty {} {} {}".format(object, variable, value), timeout
        )

    def execute_get_object_property(
        self, object: str, variable: str, timeout: float = 5
    ):
        """Get object property. \n\tUsage: GetObjectProperty [path/alias/objectname] [variable]"""
        if not (isinstance(object, str)):
            print(
                "get_object_property: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(variable, str)):
            print(
                "get_object_property: variable has wrong type. Expected str got {}".format(
                    variable.__class__.__name__
                )
            )
        self.execute("GetObjectProperty {} {}".format(object, variable), timeout)

    def request_get_object_property(
        self, object: str, variable: str, timeout: float = 5
    ):
        """Get object property. \n\tUsage: GetObjectProperty [path/alias/objectname] [variable]"""
        if not (isinstance(object, str)):
            print(
                "get_object_property: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(variable, str)):
            print(
                "get_object_property: variable has wrong type. Expected str got {}".format(
                    variable.__class__.__name__
                )
            )
        return self.request("GetObjectProperty {} {}".format(object, variable), timeout)

    def execute_change_sensor_profile(
        self, object: str, filename: str, timeout: float = 5
    ):
        """Change sensor profile for a vehicle. \n\tUsage: ChangeSensorProfile [path/alias/objectname] [filename]"""
        if not (isinstance(object, str)):
            print(
                "change_sensor_profile: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "change_sensor_profile: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        self.execute("ChangeSensorProfile {} {}".format(object, filename), timeout)

    def request_change_sensor_profile(
        self, object: str, filename: str, timeout: float = 5
    ):
        """Change sensor profile for a vehicle. \n\tUsage: ChangeSensorProfile [path/alias/objectname] [filename]"""
        if not (isinstance(object, str)):
            print(
                "change_sensor_profile: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(filename, str)):
            print(
                "change_sensor_profile: filename has wrong type. Expected str got {}".format(
                    filename.__class__.__name__
                )
            )
        return self.request(
            "ChangeSensorProfile {} {}".format(object, filename), timeout
        )

    def execute_get_object_location(self, object: str, timeout: float = 5):
        """Get the location of an object. \n\tUsage: GetObjectLocation [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_object_location: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        self.execute("GetObjectLocation {}".format(object), timeout)

    def request_get_object_location(self, object: str, timeout: float = 5):
        """Get the location of an object. \n\tUsage: GetObjectLocation [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_object_location: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        return self.request("GetObjectLocation {}".format(object), timeout)

    def request_get_object_location_alsvector(self, object: str, timeout: float = 5):
        """Get the location of an object. \n\tUsage: GetObjectLocation [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_object_location: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        response = self.request("GetObjectLocation {}".format(object), timeout)
        array = response.split(" ")
        vector = ALSVector(0, 0, 0)
        for item in array:
            key, value = item.split("=")
            if key.lower() == "x":
                vector.x = float(value)
            elif key.lower() == "y":
                vector.y = float(value)
            elif key.lower() == "z":
                vector.z = float(value)
        return vector

    def execute_get_object_rotation(self, object: str, timeout: float = 5):
        """Get the rotation of an object. \n\tUsage: GetObjectRotation [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_object_rotation: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        self.execute("GetObjectRotation {}".format(object), timeout)

    def request_get_object_rotation(self, object: str, timeout: float = 5):
        """Get the rotation of an object. \n\tUsage: GetObjectRotation [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_object_rotation: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        return self.request("GetObjectRotation {}".format(object), timeout)

    def request_get_object_rotation_alsvector_euler(
        self, object: str, timeout: float = 5
    ):
        """Get the rotation of an object. \n\tUsage: GetObjectRotation [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_object_rotation: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        response = self.request("GetObjectRotation {}".format(object), timeout)
        array = response.split(" ")
        vector = ALSVector(0, 0, 0)
        for item in array:
            key, value = item.split("=")
            if key.lower() == "r":
                vector.x = float(value)
            elif key.lower() == "p":
                vector.y = float(value)
            elif key.lower() == "y":
                vector.z = float(value)
        return vector

    def execute_get_camera_fov(
        self, ego_object: str, camera_path: str, timeout: float = 5
    ):
        """Get the vertical and horizontal FoV of the camera sensor. \n\tUsage: GetCameraFoV [Ego path/alias/objectname] [camera path]"""
        if not (isinstance(ego_object, str)):
            print(
                "get_camera_fov: ego_object has wrong type. Expected str got {}".format(
                    ego_object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "get_camera_fov: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        self.execute("GetCameraFoV {} {}".format(ego_object, camera_path), timeout)

    def request_get_camera_fov(
        self, ego_object: str, camera_path: str, timeout: float = 5
    ):
        """Get the vertical and horizontal FoV of the camera sensor. \n\tUsage: GetCameraFoV [Ego path/alias/objectname] [camera path]"""
        if not (isinstance(ego_object, str)):
            print(
                "get_camera_fov: ego_object has wrong type. Expected str got {}".format(
                    ego_object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "get_camera_fov: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        return self.request(
            "GetCameraFoV {} {}".format(ego_object, camera_path), timeout
        )

    def request_set_ptz_offset(
        self,
        object: str,
        camera_path: str,
        pan: str,
        tilt: str,
        zoom: str,
        timeout: float = 5,
    ):
        """Set the pan, tilt and zoom of a camera. \n\tUsage: SetPTZOffset [Ego path/alias/objectname] [camera path] [pan] [tilt] [zoom]"""
        if not (isinstance(object, str)):
            print(
                "set_ptz_offset: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "set_ptz_offset: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        if not (isinstance(pan, str)):
            print(
                "set_ptz_offset: pan has wrong type. Expected str got {}".format(
                    pan.__class__.__name__
                )
            )
        if not (isinstance(tilt, str)):
            print(
                "set_ptz_offset: tilt has wrong type. Expected str got {}".format(
                    tilt.__class__.__name__
                )
            )
        if not (isinstance(zoom, str)):
            print(
                "set_ptz_offset: zoom has wrong type. Expected str got {}".format(
                    zoom.__class__.__name__
                )
            )
        return self.request(
            "SetPTZOffset {} {} {} {} {}".format(object, camera_path, pan, tilt, zoom),
            timeout,
        )

    def execute_get_camera_distortion_params(
        self, object: str, camera_path: str, timeout: float = 5
    ):
        """Get the distortion parameters of the camera sensor. They are returned in the order: k1,k2,p1,p2,k3 \n\tUsage: GetCameraDistortionParams [Ego path/alias/objectname] [camera path]"""
        if not (isinstance(object, str)):
            print(
                "get_camera_distortion_params: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "get_camera_distortion_params: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        self.execute(
            "GetCameraDistortionParams {} {}".format(object, camera_path), timeout
        )

    def request_get_camera_distortion_params(
        self, object: str, camera_path: str, timeout: float = 5
    ):
        """Get the distortion parameters of the camera sensor. They are returned in the order: k1,k2,p1,p2,k3 \n\tUsage: GetCameraDistortionParams [Ego path/alias/objectname] [camera path]"""
        if not (isinstance(object, str)):
            print(
                "get_camera_distortion_params: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "get_camera_distortion_params: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        return self.request(
            "GetCameraDistortionParams {} {}".format(object, camera_path), timeout
        )

    def execute_get_lidar_info(self, object: str, camera_path: str, timeout: float = 5):
        """Returns details parameters of a specific lidar. \n\tUsage: GetLidarInfo [Ego path/alias/objectname] [camera path]"""
        if not (isinstance(object, str)):
            print(
                "get_lidar_info: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "get_lidar_info: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        self.execute("GetLidarInfo {} {}".format(object, camera_path), timeout)

    def request_get_lidar_info(self, object: str, camera_path: str, timeout: float = 5):
        """Returns details parameters of a specific lidar. \n\tUsage: GetLidarInfo [Ego path/alias/objectname] [camera path]"""
        if not (isinstance(object, str)):
            print(
                "get_lidar_info: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(camera_path, str)):
            print(
                "get_lidar_info: camera_path has wrong type. Expected str got {}".format(
                    camera_path.__class__.__name__
                )
            )
        return self.request("GetLidarInfo {} {}".format(object, camera_path), timeout)

    def execute_disable_all_shadows(self, timeout: float = 5):
        """Disables the shadows from all light sources"""
        self.execute("DisableAllShadows", timeout)

    def request_disable_all_shadows(self, timeout: float = 5):
        """Disables the shadows from all light sources"""
        return self.request("DisableAllShadows", timeout)

    def execute_enable_all_shadows(self, timeout: float = 5):
        """Enable the shadows from all light sources"""
        self.execute("EnableAllShadows", timeout)

    def request_enable_all_shadows(self, timeout: float = 5):
        """Enable the shadows from all light sources"""
        return self.request("EnableAllShadows", timeout)

    def get_sensor_list(self, timeout: float = 5):
        """Gets the sensor list"""
        return self.request("GetSensorDistributionInfo", timeout)

    def execute_get_center_of_mass(self, object: str, timeout: float = 5):
        """Get center of mass for object. \n\tUsage: GetCenterOfMass [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_center_of_mass: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        self.execute("GetCenterOfMass {}".format(object), timeout)

    def request_get_center_of_mass(self, object: str, timeout: float = 5):
        """Get center of mass for object. \n\tUsage: GetCenterOfMass [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_center_of_mass: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        return self.request("GetCenterOfMass {}".format(object), timeout)

    def execute_get_sensor_distribution_info(
        self, timeout: float = 5, include_settings: bool = False
    ):
        """Get list of sensor information. Data returned in format: 'Client ID;Sensor Path;Ego Filename;Ego Path;Ego Alias;Sensor Type;Sensor Port;Sensor IP;' and groups with the format 'ClientID;GroupName;SocketPort;SocketIP;SocketPRotocol;StreamToNetwork'"""
        self.execute(f"GetSensorDistributionInfo {include_settings}", timeout)

    def request_get_sensor_distribution_info(
        self, timeout: float = 5, include_settings: bool = False
    ):
        """Get list of sensor information. Data returned in format: 'Client ID;Sensor Path;Ego Filename;Ego Path;Ego Alias;Sensor Type;Sensor Port;Sensor IP;' and groups with the format 'ClientID;GroupName;SocketPort;SocketIP;SocketPRotocol;StreamToNetwork'"""
        return self.request(f"GetSensorDistributionInfo {include_settings}", timeout)

    def execute_get_sensor_info(
        self, object: str, sensor_path: str, timeout: float = 5
    ):
        """Returns the currently set parameters for a specific sensor. \n\tUsage: GetSensorInfo [Ego path/alias/objectname] [sensor path]"""
        if not (isinstance(object, str)):
            print(
                "get_sensor_info: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(sensor_path, str)):
            print(
                "get_sensor_info: sensor_path has wrong type. Expected str got {}".format(
                    sensor_path.__class__.__name__
                )
            )
        self.execute("GetSensorInfo {} {}".format(object, sensor_path), timeout)

    def request_get_sensor_info(
        self, object: str, sensor_path: str, timeout: float = 5
    ):
        """Returns the currently set parameters for a specific sensor. \n\tUsage: GetSensorInfo [Ego path/alias/objectname] [sensor path]"""
        if not (isinstance(object, str)):
            print(
                "get_sensor_info: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(sensor_path, str)):
            print(
                "get_sensor_info: sensor_path has wrong type. Expected str got {}".format(
                    sensor_path.__class__.__name__
                )
            )
        return self.request("GetSensorInfo {} {}".format(object, sensor_path), timeout)

    def execute_set_object_velocity(
        self,
        object: str,
        local_space: bool,
        X: float,
        Y: float,
        Z: float,
        timeout: float = 5,
    ):
        """Set object velocity in centimeters per second. \n\tUsage: SetObjectVelocity [path/alias/objectname] [local space] [X] [Y] [Z]"""
        if not (isinstance(object, str)):
            print(
                "set_object_velocity: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(local_space, bool)):
            print(
                "set_object_velocity: local_space has wrong type. Expected bool got {}".format(
                    local_space.__class__.__name__
                )
            )
        if not (isinstance(X, int) or isinstance(X, float)):
            print(
                "set_object_velocity: X has wrong type. Expected float got {}".format(
                    X.__class__.__name__
                )
            )
        if not (isinstance(Y, int) or isinstance(Y, float)):
            print(
                "set_object_velocity: Y has wrong type. Expected float got {}".format(
                    Y.__class__.__name__
                )
            )
        if not (isinstance(Z, int) or isinstance(Z, float)):
            print(
                "set_object_velocity: Z has wrong type. Expected float got {}".format(
                    Z.__class__.__name__
                )
            )
        self.execute(
            "SetObjectVelocity {} {} {} {} {}".format(object, local_space, X, Y, Z),
            timeout,
        )

    def request_set_object_velocity(
        self,
        object: str,
        local_space: bool,
        X: float,
        Y: float,
        Z: float,
        timeout: float = 5,
    ):
        """Set object velocity in centimeters per second. \n\tUsage: SetObjectVelocity [path/alias/objectname] [local space] [X] [Y] [Z]"""
        if not (isinstance(object, str)):
            print(
                "set_object_velocity: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(local_space, bool)):
            print(
                "set_object_velocity: local_space has wrong type. Expected bool got {}".format(
                    local_space.__class__.__name__
                )
            )
        if not (isinstance(X, int) or isinstance(X, float)):
            print(
                "set_object_velocity: X has wrong type. Expected float got {}".format(
                    X.__class__.__name__
                )
            )
        if not (isinstance(Y, int) or isinstance(Y, float)):
            print(
                "set_object_velocity: Y has wrong type. Expected float got {}".format(
                    Y.__class__.__name__
                )
            )
        if not (isinstance(Z, int) or isinstance(Z, float)):
            print(
                "set_object_velocity: Z has wrong type. Expected float got {}".format(
                    Z.__class__.__name__
                )
            )
        return self.request(
            "SetObjectVelocity {} {} {} {} {}".format(object, local_space, X, Y, Z),
            timeout,
        )

    def execute_set_object_angular_velocity(
        self,
        object: str,
        local_space: bool,
        X: float,
        Y: float,
        Z: float,
        timeout: float = 5,
    ):
        """Set object angular velocity in degrees per second. \n\tUsage: SetObjectAngularVelocity [path/alias/objectname] [local space] [X] [Y] [Z]"""
        if not (isinstance(object, str)):
            print(
                "set_object_angular_velocity: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(local_space, bool)):
            print(
                "set_object_angular_velocity: local_space has wrong type. Expected bool got {}".format(
                    local_space.__class__.__name__
                )
            )
        if not (isinstance(X, int) or isinstance(X, float)):
            print(
                "set_object_angular_velocity: X has wrong type. Expected float got {}".format(
                    X.__class__.__name__
                )
            )
        if not (isinstance(Y, int) or isinstance(Y, float)):
            print(
                "set_object_angular_velocity: Y has wrong type. Expected float got {}".format(
                    Y.__class__.__name__
                )
            )
        if not (isinstance(Z, int) or isinstance(Z, float)):
            print(
                "set_object_angular_velocity: Z has wrong type. Expected float got {}".format(
                    Z.__class__.__name__
                )
            )
        self.execute(
            "SetObjectAngularVelocity {} {} {} {} {}".format(
                object, local_space, X, Y, Z
            ),
            timeout,
        )

    def request_set_object_angular_velocity(
        self,
        object: str,
        local_space: bool,
        X: float,
        Y: float,
        Z: float,
        timeout: float = 5,
    ):
        """Set object angular velocity in degrees per second. \n\tUsage: SetObjectAngularVelocity [path/alias/objectname] [local space] [X] [Y] [Z]"""
        if not (isinstance(object, str)):
            print(
                "set_object_angular_velocity: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(local_space, bool)):
            print(
                "set_object_angular_velocity: local_space has wrong type. Expected bool got {}".format(
                    local_space.__class__.__name__
                )
            )
        if not (isinstance(X, int) or isinstance(X, float)):
            print(
                "set_object_angular_velocity: X has wrong type. Expected float got {}".format(
                    X.__class__.__name__
                )
            )
        if not (isinstance(Y, int) or isinstance(Y, float)):
            print(
                "set_object_angular_velocity: Y has wrong type. Expected float got {}".format(
                    Y.__class__.__name__
                )
            )
        if not (isinstance(Z, int) or isinstance(Z, float)):
            print(
                "set_object_angular_velocity: Z has wrong type. Expected float got {}".format(
                    Z.__class__.__name__
                )
            )
        return self.request(
            "SetObjectAngularVelocity {} {} {} {} {}".format(
                object, local_space, X, Y, Z
            ),
            timeout,
        )

    def execute_trigger_object(self, object: str, TriggerName: str, timeout: float = 5):
        """Call trigger on object. \n\tUsage: TriggerObject [path/alias/objectname] [TriggerName]"""
        if not (isinstance(object, str)):
            print(
                "trigger_object: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(TriggerName, str)):
            print(
                "trigger_object: TriggerName has wrong type. Expected str got {}".format(
                    TriggerName.__class__.__name__
                )
            )
        self.execute("TriggerObject {} {}".format(object, TriggerName), timeout)

    def request_trigger_object(self, object: str, TriggerName: str, timeout: float = 5):
        """Call trigger on object. \n\tUsage: TriggerObject [path/alias/objectname] [TriggerName]"""
        if not (isinstance(object, str)):
            print(
                "trigger_object: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        if not (isinstance(TriggerName, str)):
            print(
                "trigger_object: TriggerName has wrong type. Expected str got {}".format(
                    TriggerName.__class__.__name__
                )
            )
        return self.request("TriggerObject {} {}".format(object, TriggerName), timeout)

    def execute_gps_to_cartesian(
        self, latitude: float, longitude: float, timeout: float = 5
    ):
        """Converts the specified geographic coordinates to Cartesian coordinates (x, y). \n\tUsage: GPSToCartesian [latitude] [longitude]"""
        if not (isinstance(latitude, int) or isinstance(latitude, float)):
            print(
                "gps_to_cartesian: latitude has wrong type. Expected float got {}".format(
                    latitude.__class__.__name__
                )
            )
        if not (isinstance(longitude, int) or isinstance(longitude, float)):
            print(
                "gps_to_cartesian: longitude has wrong type. Expected float got {}".format(
                    longitude.__class__.__name__
                )
            )
        self.execute("GPSToCartesian {} {}".format(latitude, longitude), timeout)

    def request_gps_to_cartesian(
        self, latitude: float, longitude: float, timeout: float = 5
    ):
        """Converts the specified geographic coordinates to Cartesian coordinates (x, y). \n\tUsage: GPSToCartesian [latitude] [longitude]"""
        if not (isinstance(latitude, int) or isinstance(latitude, float)):
            print(
                "gps_to_cartesian: latitude has wrong type. Expected float got {}".format(
                    latitude.__class__.__name__
                )
            )
        if not (isinstance(longitude, int) or isinstance(longitude, float)):
            print(
                "gps_to_cartesian: longitude has wrong type. Expected float got {}".format(
                    longitude.__class__.__name__
                )
            )
        return self.request("GPSToCartesian {} {}".format(latitude, longitude), timeout)

    def execute_get_all_object_properties(self, object: str, timeout: float = 5):
        """Returns a list of all object properties. \n\tUsage: GetAllObjectProperties [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_all_object_properties: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        self.execute("GetAllObjectProperties {}".format(object), timeout)

    def request_get_all_object_properties(self, object: str, timeout: float = 5):
        """Returns a list of all object properties. \n\tUsage: GetAllObjectProperties [path/alias/objectname]"""
        if not (isinstance(object, str)):
            print(
                "get_all_object_properties: object has wrong type. Expected str got {}".format(
                    object.__class__.__name__
                )
            )
        return self.request("GetAllObjectProperties {}".format(object), timeout)

    def execute_get_object_list(self, timeout: float = 5):
        """Returns a list of all object aliases. \n\tUsage: GetObjectList"""
        self.execute("GetObjectList", timeout)

    def request_get_object_list(self, timeout: float = 5):
        """Returns a list of all object aliases. \n\tUsage: GetObjectList"""
        return self.request("GetObjectList", timeout)

    def request_load_dataasset_actor(
        self, dataAsset: str, overrideFile: str = "", timeout: float = 5
    ):
        """Load a data asset actor. \n\tUsage: LoadDataAssetActor [dataasset] [overridefile]"""
        command = f"LoadDataAssetActor {dataAsset} {overrideFile}"
        return_value = self.request(command, timeout)
        self.wait_for_task_complete()
        return return_value

    def execute_load_python_file_for_overrides(self, filename: str, timeout: float = 5):
        """Loads a file containing one or more Python functions to be used in function overrides \n\tUsage: LoadPythonFileForOverrides [file name] (in 'ConfigFiles/Overrides')"""
        if not (isinstance(filename, str) and len(filename) > 0):
            print("load_python_file_for_overrides: filename was empty or not a string")
        self.execute(f"LoadPythonFileForOverrides {filename}", timeout)

    def request_load_python_file_for_overrides(self, filename: str, timeout: float = 5):
        """Loads a file containing one or more Python functions to be used in function overrides \n\tUsage: LoadPythonFileForOverrides [file name] (in 'ConfigFiles/Overrides')"""
        if not (isinstance(filename, str) and len(filename) > 0):
            print("load_python_file_for_overrides: filename was empty or not a string")
        return self.request(f"LoadPythonFileForOverrides {filename}", timeout)
