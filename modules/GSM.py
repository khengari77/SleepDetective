import time
import serial


class GSM:
    def __init__(self, port="/dev/ttyS0", baud_rate=115200):
        self.serial = serial.Serial(port, baud_rate, timeout=1)
        self.number = "+218922653284"
        self.message = "Help"
        self.sent = False

    def send_command(self, cmd, timeout=1):
        self.serial.write(cmd.encode("utf-8") + b"\r\n")
        time.sleep(timeout)

    def send_SMS(self, number, message, timeout=2):
        self.send_command('AT+CMGF=1')
        time.sleep(timeout)
        self.send_command(f'AT+CMGS="{number}"', response_wait_time=2)
        self.serial.write(message.encode('utf-8'))
        self.serial.write(b'\x1A')  # Ctrl+Z to end SMS
        time.sleep(timeout)
