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

labels_dict = {
    'other' : ['something else'],
    'bed_to_toilet' : ['using the toilet', 'on the toilet'],
    'sleep' : ['sleeping', 'going to sleep'],
    'take_medicine' : ['taking medicine', 'taking pills', 'taking my pills'],
    'relax' : ['relaxing', 'watching TV', 'watching a movie', 'reading'],
    'cook' : ['making dinner', 'making lunch', 'making breakfast', 'cooking'],
    'work' : ['working', 'doing work'],
    'leave_home' : ['going out', 'heading out', 'going outside'],
    'bathing' : ['taking a bath', 'taking a shower', 'having a bath', 'having a shower'],
    'eat' : ['eating', 'eating lunch', 'eating dinner', 'eating breakfast', 'having lunch', 'having dinner', 'having breakfast']
}

class CASASCommitteePredict(object):
    def __init__(self, debug):
        self.id = 'committee_predict'

        self.logger = Log(self.id)

        self.debug = debug

        self.counter = 0
        self.load_test_data_and_models()
        self.load_labels()

        self.logger.log_great('Ready.')

    # Data

    def load_test_data_and_models(self):
        self.logger.log('Loading test data...')
        x_test = pd.read_csv('data/CASAS/x_test.csv', header=None)
        y_test = pd.read_csv('data/CASAS/y_test.csv', header=None)

        self.logger.log('Loading models...')
        model_LSTM = keras.models.load_model('models/CASAS/LSTM.h5')
        model_biLSTM = keras.models.load_model('models/CASAS/biLSTM.h5')
        model_CascadeLSTM = keras.models.load_model('models/CASAS/CascadeLSTM.h5')

        x_test = x_test.values
        y_test = y_test.values

        x_test = np.array(x_test)

        self.x_test = x_test
        self.y_test = y_test
        self.model_LSTM = model_LSTM
        self.model_biLSTM = model_biLSTM
        self.model_CascadeLSTM = model_CascadeLSTM

    def load_labels(self):
        labels = np.load('data/CASAS/labels.npy', allow_pickle=True)
        labels = labels.tolist()
        labels = {v: k for k, v in labels.items()} # inverse the map
        self.labels = labels
        print(labels)

    # Predicting

    def make_single_prediction(self, model, sample):
        y_pred = model.predict(sample)
        return y_pred

    def next_prediction(self):
        sample = self.x_test[self.counter]

        # Keras LSTM expects 3D tensor even if batch size is one sample
        sample = np.expand_dims(sample, axis=0)
        print('Sample:', sample)

        y_pred_LSTM = self.make_single_prediction(self.model_LSTM, sample)
        y_pred_biLSTM = self.make_single_prediction(self.model_biLSTM, sample)
        y_pred_CascadeLSTM = self.make_single_prediction(self.model_CascadeLSTM, sample)

        if self.debug:
            print('Actual:', self.y_test[self.counter], 'Predictions: LSTM =', np.argmax(y_pred_LSTM),', biLSTM =', np.argmax(y_pred_biLSTM), ', Cascade:LSTM = ', np.argmax(y_pred_CascadeLSTM))

        committee_vote_1 = y_pred_LSTM[0]
        committee_vote_2 = y_pred_biLSTM[0]
        committee_vote_3 = y_pred_CascadeLSTM[0]
        
        true = self.y_test[self.counter]
        true = true[0]

        self.counter = self.counter + 1

        return committee_vote_1, committee_vote_2, committee_vote_3, true

    # Class Methods

    def get_label(self, class_number):
        return self.labels[class_number]

    def get_labels_dict(self):
        return labels_dict

    def reset_counter(self):
        self.counter = 0

    def get_max_predictions(self):
        return len(self.y_test)