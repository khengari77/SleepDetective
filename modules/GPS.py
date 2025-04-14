import time
from threading import Thread
from serial import Serial, SerialException

class GPS:
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600):
        self.port = port
        self.serial = Serial(port, baudrate, timeout=1)
        self.baudrate = baudrate
        self.longitude = None
        self.latitude = None
        self.time_data = None
        self.location_data = None
        self.stopped = False

    def start(self):
        Thread(target=self.get).start()
        return self

    def get(self):
        while not self.stopped:
            time.sleep(0.1)
            try:
                line = self.serial.readline().decode('utf-8').strip()
                if line.startswith('$GPRMC'):
                    data = line.split(',')
                    if len(data) >= 10 and data[2] == 'A':  # Check if the GPS data is valid
                        self.parse_gps_data(data)
            except SerialException:
                print("Serial connection lost. Exiting GPS thread.")
                break

    def parse_gps_data(self, data):
        time_str = data[1][:2] + ':' + data[1][2:4] + ':' + data[1][4:6]
        self.latitude = self.convert_degrees(data[3], data[4])
        self.longitude = self.convert_degrees(data[5], data[6])

        self.datetime = time_str
        self.location_data = (self.latitude, self.longitude)

    def convert_degrees(self, value, direction):
        degrees = int(value[:2])
        minutes = float(value[2:])
        result = degrees + minutes / 60.0
        if direction == 'S' or direction == 'W':
            result = -result
        return result

    def get_location(self):
        return self.location_data

    def get_time(self):
        return self.time_data

    def stop(self):
        if self.serial_port:
            self.serial_port.close()
        self.stopped = True
