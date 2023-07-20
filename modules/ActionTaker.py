from periphery import GPIO
import time
from threading import Thread
from .GSM import GSM

class ActionTaker:

    def __init__(self):
        self.stopped = False
        self.pin_state = False
        self.last_state = False        
        self.pin = GPIO(71, "out")
        self.gsm  = GSM(port="/dev/ttyS4")
        self.message_sent = False

    def take(self, state):
        self.last_state = self.pin_state
        self.pin_state = state

    def start(self):
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
