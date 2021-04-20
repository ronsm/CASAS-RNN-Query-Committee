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
from tensorflow import keras
import sys
from time import perf_counter
from time import sleep
import pickle

from log import Log

class CASASCommitteePredict(object):
    def __init__(self, debug):
        self.id = 'committee_predict'

        self.logger = Log(self.id)

        self.debug = debug

        self.counter = 0
        self.load_test_data_and_models()
        self.load_labels()

        self.current_sample = None

        self.logger.log_great('Ready.')

    # Data

    def load_test_data_and_models(self):
        self.logger.log('Loading test data...')
        x_test = pd.read_csv('data/CASAS/CSVs/x_test.csv', header=None)
        y_test = pd.read_csv('data/CASAS/CSVs/y_test.csv', header=None)

        self.logger.log('Loading models...')
        model_1 = pickle.load(open('models/CASAS/Model1.p', 'rb'))
        model_2 = pickle.load(open('models/CASAS/Model2.p', 'rb'))
        model_3 = pickle.load(open('models/CASAS/Model3.p', 'rb'))

        x_test = x_test.values
        y_test = y_test.values

        x_test = np.array(x_test)

        self.x_test = x_test
        self.y_test = y_test
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3

    def load_labels(self):
        labels = np.load('data/CASAS/labels.npy', allow_pickle=True)
        labels = labels.tolist()
        labels = {v: k for k, v in labels.items()} # inverse the map
        self.labels = labels
        print(labels)

    # Predicting

    def make_single_prediction(self, model, sample):
        y_pred = model.predict_proba(sample)
        return y_pred

    def next_prediction(self):
        sample = self.x_test[self.counter]
        self.save_sample(sample)

        # Keras LSTM expects 3D tensor even if batch size is one sample
        sample = np.expand_dims(sample, axis=0)
        print('Sample:', sample)

        y_pred_model_1 = self.make_single_prediction(self.model_1, sample)
        y_pred_model_2 = self.make_single_prediction(self.model_2, sample)
        y_pred_model_3 = self.make_single_prediction(self.model_3, sample)

        if self.debug:
            print('Actual:', self.y_test[self.counter], 'Predictions: Model 1 =', np.argmax(y_pred_model_1),', Model 2 =', np.argmax(y_pred_model_2), ', Model 3 = ', np.argmax(y_pred_model_1))

        committee_vote_1 = y_pred_model_1[0]
        committee_vote_2 = y_pred_model_2[0]
        committee_vote_3 = y_pred_model_3[0]
        
        true = self.y_test[self.counter]
        true = true[0]

        self.counter = self.counter + 1

        return committee_vote_1, committee_vote_2, committee_vote_3, true

    def save_sample(self, sample):
        self.current_sample = sample

    # Class Methods

    def get_label(self, class_number):
        return self.labels[class_number]

    def get_inverse_label(self, label):
        key = next(key for key, value in self.labels.items() if value == label)
        return key

    def reset_counter(self):
        self.counter = 0

    def get_max_predictions(self):
        return len(self.y_test)

    def get_current_sample(self):
        return self.current_sample