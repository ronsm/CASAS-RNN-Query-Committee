#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier 
from skmultiflow.bayes import NaiveBayes
from tensorflow import keras
import sys
from time import perf_counter
from time import sleep
import pickle

from log import Log

labels = ['Other', 'Going Out', 'Preparing Breakfast', 'Having Breakfast', 'Preparing Lunch', 'Having Lunch', 'Preparing Dinner', 'Having Dinner', 'Washing Dishes',
    'Having Snack', 'Sleeping', 'Watching TV', 'Studying', 'Having Shower', 'Toileting', 'Napping', 'Using Internet', 'Reading Book', 'Laundry', 'Shaving',
    'Brushing Teeth', 'Talking on the Phone', 'Listening to Music', 'Cleaning', 'Having Conversation', 'Having Guest', 'Changing Clothes']

class ARASCommitteePredict(object):
    def __init__(self):
        self.id = 'committee_predict'

        self.logger = Log(self.id)

        self.counter = 0
        self.load_test_data_and_models()

        self.logger.log_great('Ready.')

    # Data

    def load_test_data_and_models(self):
        self.model_1 = pickle.load(open('models/ARAS/Model1.p', 'rb'))
        self.model_2 = pickle.load(open('models/ARAS/Model2.p', 'rb'))
        self.model_3 = pickle.load(open('models/ARAS/Model3.p', 'rb'))

        self.stream = FileStream('data/ARAS/test.csv')

    def make_single_prediction(self, model, X):
        y_pred = model.predict_proba(X)

        return y_pred

    def next_prediction(self):
        if self.stream.has_more_samples():
            X, y = self.stream.next_sample()
            self.counter = self.counter + 1

            y_pred_model_1 = self.make_single_prediction(self.model_1, X)
            y_pred_model_2 = self.make_single_prediction(self.model_2, X)
            y_pred_model_3 = self.make_single_prediction(self.model_3, X)

            print('Actual:', y, 'Predictions: Model 1 =', np.argmax(y_pred_model_1),', Model 2 =', np.argmax(y_pred_model_2), ', Model 3 = ', np.argmax(y_pred_model_3))

            committee_vote_1 = y_pred_model_1[0]
            committee_vote_2 = y_pred_model_2[0]
            committee_vote_3 = y_pred_model_3[0]

            true = y[0]
            return committee_vote_1, committee_vote_2, committee_vote_3, true
        else:
            self.logger.log_warn('Reached end of stream.')
            return False

    # Class Methods

    def get_label(self, class_number):
        label = labels[class_number - 1]
        return label

    def reset_counter(self):
        self.counter = 0

    def get_max_predictions(self):
        return self.stream.n_remaining_samples()