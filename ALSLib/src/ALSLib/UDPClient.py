import logging, socket, struct, time, atexit


class UDPConnectionError(Exception):
    print(Exception)
    pass


class UDPClient(object):

    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._socket = None
        self._logprefix = "(%s:%s) " % (self._host, self._port)
        atexit.register(self.disconnect)

    def connect(self, connection_attempts=10):
        connection_attempts = max(1, connection_attempts)
        error = None
        for attempt in range(1, connection_attempts + 1):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                # here we create a server listening to the port from anywhere
                # note: this is the part you need to remove if you want to connect using localhost, client only
                self._socket.bind(("0.0.0.0", self._port))
                logging.debug("%sconnected", self._logprefix)
                return
            except socket.error as exception:
                error = exception
                logging.debug(
                    "%sconnection attempt %d: %s", self._logprefix, attempt, error
                )
                time.sleep(1)
        self._reraise_exception_as_UDP_error("failed to connect", error)

    def disconnect(self):
        if self._socket is not None:
            logging.debug("%sdisconnecting", self._logprefix)
            self._socket.close()
            self._socket = None

    def connected(self):
        return self._socket is not None

    def write(self, message):
        if self._socket is None:
            logging.debug(self._logprefix + "not connected")

        try:
            self._socket.sendto(message, 0, (self._host, self._port))
        except socket.error as exception:
            self._reraise_exception_as_UDP_error("failed to write data", exception)

    def readSimple(self):
        data = None
        try:
            while True:
                self._socket.settimeout(2)
                data, addr = self._socket.recvfrom(65535)
                if len(data) < 12:
                    print(
                        "error recieving UDP packet: not enough byte for ALS Protocol"
                    )
                    break
                (messageID, packetID, message_size) = struct.unpack("<III", data[0:12])
                if packetID != 0:
                    print(
                        "not starting with first packet of the message, discarding (was ",
                        messageID,
                        " ",
                        packetID,
                        ")",
                    )
                    continue
                message = data[12:]
                while len(message) != message_size:
                    self._socket.settimeout(2)
                    data, addr = self._socket.recvfrom(65535)
                    if len(data) < 8:
                        print(
                            "error recieving UDP packet: not enough byte for ALS Protocol"
                        )
                        break
                    (newMessageID, newPacketID) = struct.unpack("<II", data[0:8])
                    if messageID != newMessageID or packetID + 1 != newPacketID:
                        print(
                            "messages are not ordered: discarding: ",
                            newPacketID,
                            "(",
                            packetID,
                            ") mesID ",
                            messageID,
                            "(",
                            newMessageID,
                            ")",
                        )
                        message = ""
                        break
                    packetID = newPacketID
                    message += data[8:]
                if len(message) == message_size:
                    return message
        except socket.error as exception:
            print("exception " + str(exception))
            # self._reraise_exception_as_UDP_error('failed to read data', exception)
        return data

    def _reraise_exception_as_UDP_error(self, message, exception):
        # pass
        raise UDPConnectionError(
            "%s:%s - %s: %s" % (self._host, self._port, message, exception)
        )
