import logging, socket, struct, time, atexit
from .ALSReceiver import alsreceiver


class TCPConnectionError(Exception):
    pass


class TCPClient(object):

    def __init__(self, host, port, timeout):
        self._host = host
        self._port = port
        self._timeout = timeout
        self.receiver = None
        self._logprefix = "(%s:%s) " % (self._host, self._port)
        self._threaded = False
        atexit.register(self.disconnect)

    def connect(self, connection_attempts=10):
        self.receiver = alsreceiver()
        connection_attempts = max(1, connection_attempts)
        for attempt in range(1, connection_attempts + 1):
            did_connect = self.receiver.connect(self._host, f"{self._port}")
            if did_connect:
                logging.debug("%sconnected", self._logprefix)
                return
            else:
                logging.debug("%sconnection attempt %d", self._logprefix, attempt)

        self.receiver = None
        raise Exception("Failed to connect")

    def set_timeouts(self, connection, receive=-1, receive_fragment=-1, send=-1):
        if self.receiver is not None:
            return self.receiver.set_timeouts(
                float(connection), float(receive), float(receive_fragment), float(send)
            )
        return False

    def disconnect(self):
        if self.receiver is not None:
            logging.debug("%sdisconnecting", self._logprefix)
            self.receiver.disconnect()
            self.receiver = None

    def connected(self):
        return self.receiver is not None

    def write(self, message):
        if not self.connected():
            raise TCPConnectionError("Cannot send, not connected")
        if isinstance(message, str):
            return self.receiver.send(message.encode("utf-8"))
        else:
            return self.receiver.send(message)

    def read(self):
        data = self.receiver.receive_get(0)
        data_b = data.tobytes()
        if not self.receiver.free(data):
            logging.debug(f"{self._logprefix}Failed to free memory")
        return data_b

    def read_view(self):
        if self._threaded:
            return self.receiver.get_threaded_data()
        else:
            return self.receiver.receive_get(0)

    def start_receive(self):
        self._threaded = True
        return self.receiver.start_receive()

    def pause_receive(self, pause: bool, reconnect: bool = False):
        return self.receiver.pause_receive(bool(pause), bool(reconnect))

    def stop_receive(self, wait: bool = True):
        return self.receiver.stop_receive(bool(wait))

    def get_thread_stats(self) -> str:
        return self.receiver.get_thread_stats()

    def get_alsreceiver_version(self) -> str:
        return self.receiver.version()

    def get_next_id(self):
        return self.receiver.get_next_id

    def free_view(self, view):
        return self.receiver.free(view)

    # legacy
    def readSimple(self):
        try:
            data = self._socket.recv(256 * 256 * 4)
        except socket.error as exception:
            self._reraise_exception_as_tcp_error("failed to read data", exception)
        if not data:
            raise TCPConnectionError(self._logprefix + "connection closed")
        return data

    def _read_n(self, length):
        if self._socket is None:
            raise TCPConnectionError(self._logprefix + "not connected")
        buf = bytes()
        while length > 0:
            try:
                data = self._socket.recv(length)
            except socket.error as exception:
                print("exception ")

                self._reraise_exception_as_tcp_error("failed to read data", exception)
            if not data:
                raise TCPConnectionError(self._logprefix + "connection closed")
            buf += data
            length -= len(data)
        return buf

    def _reraise_exception_as_tcp_error(self, message, exception):
        raise TCPConnectionError("%s%s: %s" % (self._logprefix, message, exception))
