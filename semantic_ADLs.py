#!/usr/bin/env python3

from log import Log

data = {
    'other' : ['doing something else'],
    'relaxing' : ['relaxing', 'taking a break', 'having a break', 'sitting down', 'having a rest', 'trying to relax', 'chilling out'],
    'working' : ['working', 'doing work', 'doing some work', 'getting work done'],
    'studying' : ['studying', 'studying for an exam', 'studying for a test', 'doing homework', 'going over my notes'],
    'sleeping' : ['sleeping', 'going to sleep', 'trying to sleep', 'going to bed', 'having a nap', 'taking a nap', 'having a snooze'],
    'leaving_the_house' : ['going out', 'heading out', 'going outside', 'leaving the house', 'going to the shops', 'going to the gym'],
    'bathing' : ['taking a bath', 'going for a bath', 'taking a shower', 'going for a shower', 'having a bath', 'having a shower', 'having a wash', 'washing myself'],
    'cooking' : ['cooking', 'making dinner', 'making lunch', 'making breakfast', 'cooking breakfast', 'cooking lunch', 'cooking dinner', 'making a snack', 'prearing breakfast', 'preparing lunch', 'preparing dinner'],
    'eating' : ['eating', 'eating lunch', 'eating dinner', 'eating breakfast', 'having lunch', 'having dinner', 'having breakfast', 'munching'],
    'snacking' : ['having a snack', 'eating a snack', 'having a snack attack'],
    'watching_TV' : ['watching TV', 'watching television', 'watching a programme', 'watching the news', 'watching a movie', 'watching a film', 'watching the telly'],
    'using_computer': ['on the computer', 'on my computer', 'on my laptop', 'using the computer', 'using my laptop'],
    'using_smartphone' : ['on my phone', 'using my phone', 'texting someone', 'texting', 'looking at my phone'],
    'using_internet' : ['on the internet', 'using the internet', 'browsing the internet', 'browsing the web'],
    'washing_dishes' : ['washing the dishes', 'cleaning the dishes', 'washing up', 'doing the dishes'],
    # 'showering' : ['syn:bathing'],
    'reading' : ['reading', 'reading a book', 'reading my book', 'reading a magazine', 'browsing a magazine', 'reading the newspaper', 'reading a paper', 'reading the paper'],
    'doing_laundry' : ['doing the laundry', 'doing a washing', 'doing the washing', 'doing laundry', 'doing some laundry', 'putting a wash on', 'washing clothes'],
    'shaving' : ['shaving', 'shaving my face', 'having a shave'],
    'brushing_teeth' : ['brushing my teeth', 'cleaning my teeth'],
    'talking_on_phone' : ['on the phone', 'talking to someone on the phone', 'on the phone to', 'talking to friend on the phone', 'phoning my friend', 'phoning someone', 'calling someone'],
    'listening_to_music' : ['listening to music', 'listening to the radio', 'playing music', 'listening to CD', 'have my music on'],
    'cleaning' : ['cleaning the house', 'cleaning the room', 'tidying the house', 'cleaning the bathroom', 'having a clean', 'cleaning the kitchen', 'doing the dusting', 'dusting', 'vacuuming', 'hoovering', 'hoovering the room', 'vacuuming the room'],
    'conversing' : ['chatting', 'talking to my friend', 'talking to', 'having a conversation'],
    'hosting_guests' : ['got a friend round', 'have a friend round', 'we are having a party', 'got someone round', 'friend is here', 'have people over', 'have people round'],
    'getting_dressed' : ['getting dressed', 'changing clothes', 'getting changed', 'putting clothes on', 'putting my clothes on', 'putting my outfit on'],
    'tidying_up' : ['tidying up', 'tidying up the house', 'having a clear up', 'putting out the bins', 'having a tidy'],
    'exercising' : ['working out', 'exercising', 'cycling', 'doing my exercises', 'on the treadmill', 'lifting weights', 'doing a fitness class'],
    'hobby' : ['playing cards'],
    'taking_medication' : ['taking medicine', 'taking pills', 'taking my pills', 'taking my medication', 'taking my meds', 'taking the pill', 'taking my tablets'],
    'toileting' : ['using the toilet', 'on the toilet', 'having a pee', 'going to the bathroom', 'going to the loo', 'nipping to the loo', 'going for a pee'],
}

class SemanticADLs(object):
    def __init__(self):
        self.id = 'semantic_ADLs'

        self.logger = Log(self.id)

        self.logger.log_great('Ready.')

    def get_semantic_ADLs(self):
        return data