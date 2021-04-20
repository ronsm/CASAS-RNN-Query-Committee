#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
import colorama
from colorama import Fore, Style
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys
import os.path
import pickle

from log import Log
import CASAS_data

seed = 7
np.random.seed(seed)

class CASASALTools(object):
    def __init__(self):
        self.id = 'CASAS_AL_Tools'

        self.dataset_select = "milan"

        self.logger = Log(self.id)

        self.logger.log_great('Ready.')

    def create_train_test_csvs(self, annotations=None):
        self.logger.log('Creating train/test CSVs...')

        X, Y, dictActivities = CASAS_data.getData(self.dataset_select)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, train_size=200, random_state=seed)

        x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, shuffle=False, test_size=400, random_state=seed)

        if annotations != None:
            self.logger.log_warn('[WARNING] Annotations file is present. Annotations will be appended to the training set.')
            annotations = pd.read_csv(annotations, skiprows=1, header=None)
            y_annotations = annotations.iloc[:,-1:]
            x_annotations = annotations.drop(annotations.columns[-1], axis=1)

            x_train = pd.DataFrame(x_train)
            x_train = pd.concat([x_train, x_annotations])
            x_train = x_train.values

            y_train = pd.DataFrame(y_train)
            y_annotations = y_annotations.rename(columns={2000:0})
            y_train = pd.concat([y_train, y_annotations])
            y_train = y_train.values

        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)

        x_validation = pd.DataFrame(x_validation)
        y_validation = pd.DataFrame(y_validation)

        x_train_write = pd.DataFrame(x_train)
        y_train_write = pd.DataFrame(y_train)

        x_train_write.to_csv('data/CASAS/CSVs/x_train.csv', index=False, header=False)
        y_train_write.to_csv('data/CASAS/CSVs/y_train.csv', index=False, header=False)

        x_test.to_csv('data/CASAS/CSVs/x_test.csv', index=False, header=False)
        y_test.to_csv('data/CASAS/CSVs/y_test.csv', index=False, header=False)
        
        x_validation.to_csv('data/CASAS/CSVs/x_validation.csv', index=False, header=False)
        y_validation.to_csv('data/CASAS/CSVs/y_validation.csv', index=False, header=False)

        y_train = y_train.astype('int')

        return x_train, x_test, x_validation, y_train, y_test, y_validation, dictActivities

    def save_model(self, model, name):
        self.logger.log('Saving model...')
        save_file = 'models/CASAS/' + name + '.p'
        pickle.dump(model, open(save_file, "wb"))

    # Training

    def train_model_1(self, X, Y, dictActivities):
        self.logger.log('Training model 1...')
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

        rf = RandomForestClassifier(n_estimators=100)

        scores = cross_val_score(rf, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(scores)

        rf = rf.fit(X, Y)

        return rf

    def train_model_2(self, X, Y, dictActivities):
        self.logger.log('Training model 2...')
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

        scores = cross_val_score(gbc, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(scores)

        gbc.fit(X, Y)

        return gbc

    def train_model_3(self, X, Y, dictActivities):
        self.logger.log('Training model 3...')
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

        dt = tree.DecisionTreeClassifier()

        scores = cross_val_score(dt, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(scores)

        dt = dt.fit(X, Y)

        return dt

    def train_models(self, x_train, y_train, dictActivities):
        self.logger.log('Training all models...')

        model_1 = self.train_model_1(x_train, y_train, dictActivities)
        self.save_model(model_1, 'Model1')

        model_2 = self.train_model_2(x_train, y_train, dictActivities)
        self.save_model(model_2, 'Model2')

        model_3 = self.train_model_3(x_train, y_train, dictActivities)
        self.save_model(model_3, 'Model3')

    def load_test_data_and_models(self):
        self.logger.log('Loading test data...')
        x_validation = pd.read_csv('data/CASAS/CSVs/x_validation.csv', header=None)
        y_validation = pd.read_csv('data/CASAS/CSVs/y_validation.csv', header=None)

        self.logger.log('Loading models...')
        model_1 = pickle.load(open('models/CASAS/Model1.p', 'rb'))
        model_2 = pickle.load(open('models/CASAS/Model2.p', 'rb'))
        model_3 = pickle.load(open('models/CASAS/Model3.p', 'rb'))

        return x_validation, y_validation, model_1, model_2, model_3

    def make_batch_predictions(self, x_test, y_test, model_1, model_2, model_3):
        x_test_data = x_test.values
        y_test_data = y_test.values

        x_test_data = np.array(x_test_data)

        model_1_preds = model_1.predict(x_test_data)
        cr_model_1 = classification_report(y_test_data, model_1_preds)
        print(cr_model_1)

        model_2_preds = model_2.predict(x_test_data)
        cr_model_2 = classification_report(y_test_data, model_2_preds)
        print(cr_model_2)

        model_3_preds = model_3.predict(x_test_data)
        cr_model_3 = classification_report(y_test_data, model_3_preds)
        print(cr_model_3)
    
    # Class Methods

    def init(self):
        x_train, x_test, x_validation, y_train, y_test, y_validation, dictActivities = self.create_train_test_csvs()
        self.train_models(x_train, y_train, dictActivities)
        x_validation, y_validation, model_1, model_2, model_3 = self.load_test_data_and_models()
        self.make_batch_predictions(x_validation, y_validation, model_1, model_2, model_3)

    def update(self, annotations):
        x_train, x_test, x_validation, y_train, y_test, y_validation, dictActivities = self.create_train_test_csvs(annotations=annotations)
        self.train_models(x_train, y_train, dictActivities)
        x_validation, y_validation, model_1, model_2, model_3 = self.load_test_data_and_models()
        self.make_batch_predictions(x_validation, y_validation, model_1, model_2, model_3)