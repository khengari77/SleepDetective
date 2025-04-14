import time
import pynmea2
from threading import Thread
from serial import Serial, SerialException

class GPS:
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600):
        """
        Initialize GPS module.
        
        Args:
            port (str): Serial port (default: '/dev/ttyAMA0')
            baudrate (int): Baud rate (default: 9600)
        """
        self.port = port
        self.baudrate = baudrate
        try:
            self.serial = Serial(port, baudrate, timeout=1)
        except SerialException as e:
            raise SerialException(f"Failed to connect to {port}: {str(e)}")
        
        self._latitude = None
        self._longitude = None
        self._time_data = None
        self._location_data = None
        self.stopped = False
        self.thread = None

    def start(self):
        """Start GPS data reading in a separate thread."""
        if self.thread is None or not self.thread.is_alive():
            self.stopped = False
            self.thread = Thread(target=self._read_gps)
            self.thread.daemon = True
            self.thread.start()
        return self

    def _read_gps(self):
        """Internal method to read and parse GPS data."""
        while not self.stopped:
            try:
                line = self.serial.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        if isinstance(msg, pynmea2.types.talker.RMC):
                            self._parse_rmc(msg)
                    except pynmea2.ParseError:
                        continue
            except SerialException:
                print("Serial connection lost. Stopping GPS thread.")
                self.stop()
                break
            except UnicodeDecodeError:
                continue
            time.sleep(0.01)

    def _parse_rmc(self, msg):
        """
        Parse RMC (Recommended Minimum) sentence.
        
        Args:
            msg: Parsed pynmea2 RMC message
        """
        if msg.status == 'A':  # Valid fix
            self._latitude = msg.latitude
            self._longitude = msg.longitude
            self._location_data = (self._latitude, self._longitude)
            if msg.datetime:
                self._time_data = msg.datetime.strftime('%H:%M:%S')

    @property
    def location(self):
        """
        Get current location.
        
        Returns:
            tuple: (latitude, longitude) or None if no valid data
        """
        return self._location_data

    @property
    def time(self):
        """
        Get current GPS time.
        
        Returns:
            str: Time in HH:MM:SS format or None if no valid data
        """
        return self._time_data

    def stop(self):
        """Stop GPS reading and close serial connection."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()

    def __enter__(self):
        """Context manager support for 'with' statement."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup when used in 'with' statement."""
        self.stop()

