from periphery import GPIO
import time
from threading import Thread
from .GSM import GSM

class ActionTaker:

    def __init__(self, use_gpio=True, use_gsm=True, gps_port="/dev/ttyS4"):
        self.stopped = False
        self.pin1_state = False
        self.pin2_state = False
        self.message_sent = False
        self.gps_port = gps_port 
        self.use_gpio = use_gpio
        self.use_gsm = use_gsm

    def take(self, awareness_level):
        self.pin1_state = awareness_level < 0.8
        self.pin2_state = awareness_level < 0.7

    def start(self):
        if self.use_gpio:
            self.pin1 = GPIO("/dev/gpiochip0", 17, "out")
            self.pin2 = GPIO("/dev/gpiochip0", 16, "out")
        if self.use_gsm:
            self.gsm  = GSM(port=self.gps_port)
        Thread(target=self.action, args=()).start()
        return self

    def action(self):
        while not self.stopped:
            self.pin1.write(self.pin1_state)
    def stop(self):
        self.pin.write(False)
        self.stopped = True
