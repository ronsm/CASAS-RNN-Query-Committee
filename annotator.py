#!/usr/bin/env python3

import numpy as np
import pandas as pd
from time import perf_counter, sleep, strftime
import csv

from log import Log

ROLLING_WINDOW = 30

class Annotator(object):
    def __init__(self, debug):
        self.id = 'annotator'

        self.logger = Log(self.id)

        self.debug = debug

        self.buffer_initiated = False

        self.buffer_lock = False

        self.logger.log_great('Ready.')

    def create_buffers(self, sample):
        self.sample_buffer = np.zeros((ROLLING_WINDOW, len(sample)))
        self.buffer_initiated = True

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

    def add_sample(self, sample):
        if not self.buffer_lock:
            if not self.buffer_initiated:
                self.create_buffers(sample)
                self.create_csv(sample)

            insert = np.asarray(sample)
            self.sample_buffer = np.vstack([self.sample_buffer, insert])
            self.sample_buffer = np.delete(self.sample_buffer, 0, 0)

    def lock_buffer(self):
        self.buffer_lock = True

    def unlock_buffer(self):
        self.buffer_lock = False

    def annotate_buffer(self, label):
        buffer = self.sample_buffer.tolist()

        for row in buffer:
            row.append(label)

        with open(self.csv_filename, 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerows(buffer)