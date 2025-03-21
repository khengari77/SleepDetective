from periphery import GPIO
import time
from threading import Thread
from .GSM import GSM

class ActionTaker:

    def __init__(self, use_gpio=True, use_gsm=True, gps_port="/dev/ttyS4"):
        self.stopped = False
        self.pin_state = False
        self.last_state = False        
        self.message_sent = False
        self.gps_port = gps_port 
        self.use_gpio = use_gpio
        self.use_gsm = use_gsm

    def take(self, state):
        self.last_state = self.pin_state
        self.pin_state = state

    def start(self):
        if self.use_gpio:
            self.pin = GPIO("/dev/gpiochip0", 17, "out")
        if self.use_gsm:
            self.gsm  = GSM(port=self.gps_port)
        Thread(target=self.action, args=()).start()
        return self

    def action(self):
        while not self.stopped:
            if self.last_state and not self.pin_state:
                time.sleep(5)
            elif self.last_state and self.pin_state and not self.message_sent:
                self.gsm.send_SMS("+218944682876", "HELP!")
            else:
                self.message_sent = False
            self.pin.write(self.pin_state)
    def stop(self):
        self.pin.write(False)
        self.stopped = True
