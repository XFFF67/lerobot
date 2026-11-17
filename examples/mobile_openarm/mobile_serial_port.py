"""Small Linux 115200/8N1 transport using the standard library; no pyserial dependency."""

import array
import fcntl
import os
import select
import termios
import time


class SerialPort:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            settings = termios.tcgetattr(self.fd)
            settings[:6] = [
                0,
                0,
                termios.CS8 | termios.CREAD | termios.CLOCAL,
                0,
                termios.B115200,
                termios.B115200,
            ]
            settings[6][termios.VMIN] = settings[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, settings)
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            if select.select([self.fd], [], [], 0)[0]:
                data = os.read(self.fd, 4096)
                if not data:
                    raise OSError("Serial device disconnected")
                return data
        except BlockingIOError:
            pass
        return b""

    @property
    def out_waiting(self):
        value = array.array("i", [0])
        fcntl.ioctl(self.fd, termios.TIOCOUTQ, value, True)
        return value[0]

    def write(self, data):
        deadline = time.monotonic() + 0.05
        sent = 0
        while sent < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                raise TimeoutError("Serial write exceeded 50 ms")
            try:
                written = os.write(self.fd, data[sent:])
            except BlockingIOError:
                continue
            if written <= 0:
                raise OSError("Serial write failed")
            sent += written
        return sent

    def reset_output_buffer(self):
        termios.tcflush(self.fd, termios.TCOFLUSH)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
