#!/usr/bin/env python3

import spacy
import pprint
import numpy as np

from semantic_ADLs import SemanticADLs

nlp_eng = spacy.load('en_core_web_lg')

SIMILARITY_MARGIN = 0.1

train = {
    'other' : ['doing something else', 'other'],
    'relaxing' : ['relaxing', 'sitting doing nothing', 'pottering about in the garden', 'chilling', 'taking a break', 'sitting down', 'chilling out', 'having a lie down', 'on a break'],
    'working' : ['doing work', 'sitting at my desk', 'doing paperwork', 'doing bookkeeping', 'doing some work', 'working at my desk'],
    'studying' : ['studying', 'studying for an exam', 'studying for a test', 'doing some homework', 'going over my notes', 'reading my study notes'],
    'sleeping' : ['trying to sleep', 'tossing and turning', 'trying to fall asleep', 'trying to get comfortable', 'going to bed', 'recharging my batteries', 'going to bed', 'taking a nap', 'having a snooze'],
    'leaving_the_house' : ['going out', 'going to the post office', 'visiting friends', 'heading out for a while', 'leaving', 'going out to work in the garden', 'going for a walk', 'getting some fresh air', 'going outside', 'leaving the house', 'getting out of the house', 'nipping out', 'popping out'],
    'bathing' : ['having a bath', 'going for a bath', 'relaxing in bath', 'taking a wash', 'taking a bath', 'washing myself', 'having a wash'],
    'cooking' : ['cooking', 'making something to eat' 'going to make dinner', 'making dinner', 'making food', 'getting some food', 'making lunch', 'making breakfast', 'cooking breakfast', 'making a snack', ' preparing dinner', 'doing some meal prep'],
    'prepare_drink' : ['making a drink', 'having a drink of juice', 'making a cup of tea', 'preparing a cup of tea', 'preparing a cup of coffee', 'making tea', 'making coffee', 'grabbing a drink'],
    'eating' : ['eating food', 'having a meal', 'eating breakfast', 'eating lunch', 'having lunch', 'stuffing my face', 'having dinner', 'munching'],
    'snacking' : ['grabbing a snack', 'having a treat', 'having a biscuit', 'having some fruit', 'having some scran', 'having a munch', 'having some chocolate', 'eating some sweeties', 'having a snack attack'],
    'watching_TV' : ['watching TV', 'sitting watching a film', 'sitting watching TV', 'relaxing watching TV', 'watching television', 'watching a programme', 'watching a film', 'watching a movie', 'telly'],
    'using_computer' : ['on the computer', 'sitting on my laptop', 'going to use my laptop', 'spending time on the PC', 'using my PC', 'working on my PC', 'on my computer', 'on my laptop'],
    'using_smartphone' : ['on the phone', 'mucking about on my phone', 'making a call', 'looking at my phone', 'browsing on my phone', 'browsing Facebook', 'scrolling Twitter', 'scrolling Instagram', 'scrolling Facebook', 'browsing Instagram', 'texting someone', 'texting'],
    'using_internet' : ['on the internet', 'shopping online', 'browsing the internet', 'looking something up on the internet', 'browsing the web', 'looking at something on the internet'],
    'washing_dishes' : ['doing dishes', 'washing dishes', 'doing the washing up', 'have to do the dishes', 'washing the dishes', 'washing up'],
    'showering' : ['showering', 'taking a shower', 'getting showered', 'having a shower', 'getting showered'],
    'reading' : ['reading', 'getting into a book', 'reading my kindle', 'reading my book', 'reading a magazine', 'browsing a magazine', 'perusing a newpaper', 'reading a paper', 'read a book', 'reading a book'],
    'doing_laundry' : ['doing laundry', 'doing a load of washing', 'loading the washing machine', 'doing the laundry', 'doing a washing', 'putting a wash on', 'washing clothes'],
    'shaving' : ['shaving', 'shaving my legs', 'shaving my armpits', 'having a shave', 'having a trim', 'having a shave'],
    'brushing_teeth' : ['brushing teeth', 'cleaning my teeth', 'brush my teeth'],
    'talking_on_phone' : ['on the phone', 'yapping on the phone', 'on the phone with someone', 'on a call', 'having a phone call', 'talking to someone on the phone', 'talking to my friend on the phone', 'phoning someone'],
    'listening_to_music' : ['listening to music', 'enjoying some music', 'putting some tunes on', 'listening to the radio', 'playing a CD', 'listening to a record', 'listening to Spotify', 'got my headphones on'],
    'cleaning' : ['cleaning', 'gutting the house', 'dusting', 'washing floors', 'doing the housework', 'cleaning up', 'sorting the place out', 'cleaning the house', 'cleaning my room', 'tidying the house', 'cleaning the bathroom', 'vacuuming', 'hoovering up'],
    'conversing' : ['having a chat', 'chatting to the neighbours', 'talking', 'yapping', 'talking to friends', 'talking to someone', 'talking to my friend', 'having a debate'],
    'hosting_guests' : ['having people over', 'have visitors', 'have friends over', 'friends round', 'hanging out', 'hanging out with friends', 'have a friend round', 'having a party', 'having a dinner party', 'hosting a dinner party'],
    'getting_dressed' : ['getting dressed', 'getting ready', 'putting my clothes on', 'putting clothes on', 'get ready', 'getting undressed', 'getting changed', 'changing clothes', 'putting my outfit on', 'getting dressed up', 'putting my makeup on'],
    'taking_medication' : ['taking pills', 'taking some pills', 'swallowing some pills', 'taking my tablets', 'taking my pills', 'taking medication', 'taking the pill', 'taking medicine'],
    'tidying_up' : ['tidying up', 'doing a quick tidy', 'tidy up', 'getting organised', 'organising the house', 'tidying the house', 'having a clear up', 'taking out the bins'],
    'toileting' : ['using the toilet', 'on the loo', 'having a wee', 'going to the toilet', 'going to the loo', 'having a piss', 'going to the loo', 'having a poo', 'taking a dump', 'doing a poo', 'having a pee', 'nipping to the loo', 'going for a pee', 'going to the bathroom', 'using the bathroom'],
    'exercising' : ['exercising', 'walking the dog', 'having a walk', 'working out', 'going to a gym class', 'doing my exercises', 'on the treadmill', 'lifting weights', 'doing a fitness class'],
    'drinking' : ['having a drink', 'having a cuppa', 'drinking my tea', 'drinking my coffee', 'drink some water', 'taking a drink', 'warming myself up with a hot drink', 'having a drink of water'],
    'hobby' : ['doing a hobby', 'having some downtime', 'building my jigsaw', 'doing a jigsaw', 'mucking about', 'doing some DIY', 'playing cards', 'playing board games', 'painting', 'playing chess', 'writing', 'pottering about in the garden', 'working in the garden', 'working in the garden'],
}

test = {
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

def compute_similarity(compare):
    all_similarity_scores = {}
    compare = nlp_eng(compare)
    
    for key, value in train.items():
        class_descriptions = []
        similarity_scores = []

        for item in value:
            class_descriptions.append(nlp_eng(item))
        
        for class_description in class_descriptions:
            similarity_score = class_description.similarity(compare)
            similarity_scores.append(similarity_score)

        all_similarity_scores[key] = similarity_scores

    return all_similarity_scores

def sort_similarity_scores(similarity_scores):
    similarity_scores_argmax = {}

    for key, value in similarity_scores.items():
        np_array = np.asarray(value)
        argmax = np_array[np_array.argmax()]

        similarity_scores_argmax[key] = argmax

    data_sorted = {k: v for k, v in sorted(similarity_scores_argmax.items(), reverse=True, key=lambda x: x[1])}

    top_label = list(data_sorted)[0]

    # print(data_sorted)

    return data_sorted, top_label

def evaluate_follow_up(similarity_scores):
    follow_up = False

    key_1 = list(similarity_scores)[0]
    key_2 = list(similarity_scores)[1]

    value_1 = list(similarity_scores.values())[0]
    value_2 = list(similarity_scores.values())[1]

    options = [key_1, key_2]

    margin = value_2 - value_1
    if margin < 0:
        margin = margin * -1.0

    if margin < SIMILARITY_MARGIN:
        follow_up = True

    return follow_up, options

count = 0
top_correct = 0
top_correct_no_follow_up = 0
options_correct = 0
for key, value in test.items():
    for item in value:
        similarity_scores = compute_similarity(item)
        similarity_scores_sorted, top_label = sort_similarity_scores(similarity_scores)
        follow_up, options = evaluate_follow_up(similarity_scores_sorted)

        print(item, '->', top_label, 'options:', options)
        # print(similarity_scores_sorted)

        if follow_up == False:
            if key == top_label:
                top_correct = top_correct + 1
        else:
            if key == options[0] or key == options[1]:
                options_correct = options_correct + 1

        if key == top_label:
            top_correct_no_follow_up = top_correct_no_follow_up + 1

        # if key == top_label:
        #     correct = correct + 1
        
        # if key == options[0] or key == options[1]:
        #     correct = correct + 1
        
        count = count + 1

        print('Top Correct (NFU):', (top_correct_no_follow_up/count), 'Top Correct (FU)', (top_correct/count), 'Options Correct:', (options_correct/count), 'Effectivepy Correct:', ((top_correct+options_correct)/count))