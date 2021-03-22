#!/usr/bin/env python3

import numpy as np
import pandas as pd
from time import perf_counter, sleep, strftime
import csv

from log import Log

class CASASAnnotator(object):
    def __init__(self, debug, dataset, committee_predict):
        self.id = 'annotator'

        self.logger = Log(self.id)

        self.debug = debug
        self.dataset = dataset
        self.committee_predict = committee_predict

        self.csv_created = False

        self.buffer_lock = False

        self.logger.log_great('Ready.')

    def create_csv(self, sample):
        date_time = strftime("%Y%m%d-%H%M%S")
        self.csv_filename = 'annotations/annotation_' + date_time + '.csv'

        msg = 'The annotation file for this session is ' + self.csv_filename
        self.logger.log(msg)

        with open(self.csv_filename, 'w', newline='') as fd:
            writer = csv.writer(fd)

            header = []
            for i in range(0, len(sample)):
                header.append(i)
            header.append('label')

            writer.writerow(header)

        self.csv_created = True

    def add_sample(self, sample):
        if not self.csv_created:
            self.create_csv(sample)

        self.hold_sample = sample

    def lock_buffer(self):
        self.buffer_lock = True

    def unlock_buffer(self):
        self.buffer_lock = False

    def annotate_buffer(self, label):
        msg = 'Annotating buffer with label: ' + label
        self.logger.log(msg)

        label = label.lower()
        label = self.committee_predict.get_inverse_label(label)

        row = self.hold_sample.tolist()
        row.append(label)

        with open(self.csv_filename, 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(row)
    
    def get_annotation_filename(self):
        return self.csv_filename