
import numpy as np
from log import Log

data = {
    'other' : ['other'],
    'relaxing' : ['trying to relax', 'having a break', 'having a rest', 'lying down'],
    'working' : ['working on the computer', 'answering phone calls', 'getting work done'],
    'studying' : ['taking time to study', 'doing homework'],
    'sleeping' : ['laying awake', 'going to sleep', 'having a nap', 'having a snooze'],
    'leaving_the_house' : ['going to the shops', 'heading out', 'going out to work in the garden', 'going to the shops', 'going to the gym'],
    'bathing' : ['having a bath', 'enjoying a bath'],
    'cooking' : ['making the dinner', 'making a meal', 'cooking lunch', 'preparing breakfast', 'preparing lunch'],
    'prepare_drink' : ['making a cup of coffee', 'making a mug of tea'],
    'eating' : ['feeding myself', 'having some food', 'eating dinner', 'having breakfast'],
    'snacking' : ['snacking', 'having a snack', 'eating a snack', 'having some sweeties'],
    'watching_TV' : ['watching telly', 'watching', 'watching the news'],
    'using_computer' : ['sitting on the computer', 'shopping on the computer', 'doing online banking', 'working on my laptop'],
    'using_smartphone' : ['on my phone', 'messaging someone', 'browsing Twitter', 'scrolling on my phone', 'checking something on my phone'],
    'using_internet' : ['online shopping', 'using the internet'],
    'washing_dishes' : ['doing the dishes', 'cleaning the dishes', 'washing up after dinner'],
    'showering' : ['having a shower', 'going for a shower'],
    'reading' : ['into my book', 'reading a book', 'reading on my kindle', 'reading the paper'],
    'doing_laundry' : ['putting washing in the washing machine', 'doing the washing', 'doing some laundry'],
    'shaving' : ['shaving my face', 'shaving my underarms'],
    'brushing_teeth' : ['brushing my teeth'],
    'talking_on_phone' : ['having a blether on the phone', 'talking on the phone', 'calling someone'],
    'listening_to_music' : ['listening to music', 'playing a record', 'listening to a CD', 'have my music on'],
    'cleaning' : ['hoovering', 'tidying up', 'spring cleaning', 'having a clean', 'cleaning the kitchen', 'doing the dusting'],
    'conversing' : ['talking to someone', 'having a conversation', 'chatting', 'debating something'],
    'hosting_guests' : ['having friends round', 'have people oever', 'got a friend round', 'hosting a party', 'have people round'],
    'getting_dressed' : ['getting dressed', 'putting makeup on', 'getting ready for bed', 'changing my clothes'],
    'taking_medication' : ['taking tablets', 'having some medication', 'taking my meds'],
    'tidying_up' : ['tidying around the house', 'tidying', 'putting out the bins'],
    'toileting' : ['on the toilet', 'having a slash', 'having a shit', 'on the pan', 'going for a pee'],
    'exercising' : ['going for a walk', 'getting some exercise', 'lifting weights'],
    'drinking' : ['drinking some water', 'drinking', 'having a drink'],
    'hobby' : ['building a jigsaw', 'doing my DIY', 'drawing', 'writing a novel', 'doing the garden', 'doing some gardenning']
}

class HumanResponseSimulator(object):
    def __init__(self, label_linker):
        self.id = 'human_response_simulator'

        self.logger = Log(self.id)

        self.label_linker = label_linker

        self.logger.log_great('Ready.')

    def get_input(self, true, follow_up, options):
        rand = np.random.uniform(low=0.0, high=1.0, size=None)

        string = 'I am ' + true

        select = -1

        if follow_up:
            model_options = []

            model_options.append(self.label_linker.get_model_label(options[0]))
            model_options.append(self.label_linker.get_model_label(options[1]))
                
            if model_options[0] == true:
                select = 0
            elif model_options[1] == true:
                select = 1

            if select != -1:
                string = 'I am ' + options[select]

        msg = 'HSR: ' + string
        self.logger.log_great(msg)

        return string