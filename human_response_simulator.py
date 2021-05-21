
import numpy as np
from log import Log
from semantic_ADLs import SemanticADLs

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

        self.semantic_ADLs = SemanticADLs()

        self.logger.log_great('Ready.')

    def get_input(self, true, follow_up, options):
        descriptor = self.get_ADL_descriptor(true, follow_up, options)

        response = self.wrap_descriptor(descriptor)
        
        self.logger.log_great(response)

        return response

    def get_ADL_descriptor(self, true, follow_up, options):
        select = -1

        descriptor = ''

        ADLs_for_true = self.label_linker.get_ADL_labels(true)
        selected_ADL = np.random.choice(ADLs_for_true)

        descriptors = data[selected_ADL]
        descriptor = np.random.choice(descriptors)

        if follow_up:
            ADL_labels = []
            model_options = []

            ADL_labels.append(self.semantic_ADLs.get_ADL_from_descriptor(options[0]))
            ADL_labels.append(self.semantic_ADLs.get_ADL_from_descriptor(options[1]))

            model_options.append(self.label_linker.get_model_label(ADL_labels[0]))
            model_options.append(self.label_linker.get_model_label(ADL_labels[1]))
                
            if model_options[0] == true:
                select = 0
            elif model_options[1] == true:
                select = 1

            if select != -1:
                descriptor = options[select]

        return descriptor

    def wrap_descriptor(self, descriptor):
        wrappings = []

        wrapping = "I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "I'm just " + descriptor
        wrappings.append(wrapping)

        wrapping = "I am just " + descriptor
        wrappings.append(wrapping)

        wrapping = "Am just " + descriptor
        wrappings.append(wrapping)

        wrapping = "No, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "No, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yes, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yes, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "I'm just going to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I am just going to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I'm just about to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I am just about to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I was about to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I was just about to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I'm still " + descriptor
        wrappings.append(wrapping)

        wrapping = "I am still " + descriptor
        wrappings.append(wrapping)

        wrapping = "I need to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I have to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I've got to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I have got to " + descriptor
        wrappings.append(wrapping)

        wrapping = "I got to " + descriptor
        wrappings.append(wrapping)

        wrapping = "At the moment I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "At the moment I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "You are correct, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "You are correct, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "That's correct, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "That's correct, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "That is correct, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "That is correct, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "That's right, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "That's right, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "That is right, I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "That is right, I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "Now I am " + descriptor
        wrappings.append(wrapping)

        wrapping = "Now I'm " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yup, still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yeah, still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yes, still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yep, still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yeah, I'm still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yeah, I am still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yes, I am still " + descriptor
        wrappings.append(wrapping)

        wrapping = "Yes, I'm still " + descriptor
        wrappings.append(wrapping)

        wrapping = "No, I am still " + descriptor
        wrappings.append(wrapping)

        wrapping = "No, I'm still " + descriptor
        wrappings.append(wrapping)

        wrapping = np.random.choice(wrappings)
        return wrapping