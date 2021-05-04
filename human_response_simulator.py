
import numpy as np

class HumanResponseSimulator(object):
    def __init__(self):
        self.id = 'human_response_simulator'

    def get_input(self, true):
        print(true)
        string = 'I am ' + true
        return string

    # TODO: Responses for each ADL