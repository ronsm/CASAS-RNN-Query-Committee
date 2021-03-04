#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from time import perf_counter, sleep, strftime
import csv
from modAL.disagreement import vote_entropy, consensus_entropy, entropy, max_disagreement_sampling

from log import Log

ROLLING_WINDOW = 30
NUM_LEARNERS = 3
QUERY_LOCK_LENGTH = 60

THRESHOLD_MAX_DISAGREEMENT_INDIVIDUAL = 0.4
THRESHOLD_MAX_DISAGREEMENT_WINDOW = 0.2
THRESHOLD_PERCENT_OF_WINDOW = 0.2 # percent as floating point number

class QuerySelect(object):
    def __init__(self, debug):
        self.id = 'query_select'
        
        self.logger = Log(self.id)

        self.debug = debug

        self.create_buffers()

        self.time_since_last_query = QUERY_LOCK_LENGTH

        self.max_disagreement = np.zeros((ROLLING_WINDOW, 1))

        self.logger.log_great('Ready.')

    # Buffer

    def create_buffers(self):
        self.committee_member_1_buffer = []
        self.committee_member_2_buffer = []
        self.committee_member_3_buffer = []
        self.true_buffer = []

        self.tail = -1
        self.head = -1

    def update_buffers(self):
        length = len(self.committee_member_1_buffer)

        if length == ROLLING_WINDOW:
            self.tail = 0
            self.head = ROLLING_WINDOW - 1

        if length > ROLLING_WINDOW:
            self.tail = self.tail + 1
            self.head = self.head + 1 

    def insert_to_buffers(self, committee_vote_1, committee_vote_2, committee_vote_3, true):
        self.committee_member_1_buffer.append(committee_vote_1)
        self.committee_member_2_buffer.append(committee_vote_2)
        self.committee_member_3_buffer.append(committee_vote_3)
        self.true_buffer.append(true)

        self.update_buffers()

    # Query Selection
    
    def calculate_max_disagreement(self):
        if self.head == -1:
            self.logger.log_warn('There are not sufficient samples in the buffer. Cannot evaluate the window.')
            return np.zeros((ROLLING_WINDOW, 1))

        committee_member_1_pred_window = self.committee_member_1_buffer[self.tail:]
        committee_member_2_pred_window = self.committee_member_2_buffer[self.tail:]
        committee_member_3_pred_window = self.committee_member_3_buffer[self.tail:]
        true_window = self.true_buffer[self.tail:]

        committee_member_1_pred_window = np.array(committee_member_1_pred_window)
        committee_member_2_pred_window = np.array(committee_member_2_pred_window)
        committee_member_3_pred_window = np.array(committee_member_3_pred_window)

        committee_merged_pred_window = np.array([committee_member_1_pred_window, committee_member_2_pred_window, committee_member_3_pred_window])

        consensus_prob = np.mean([committee_member_1_pred_window, committee_member_2_pred_window, committee_member_3_pred_window], axis=0)

        if self.debug:
            self.logger.log_math('Consensus Probabilities:')
            print(consensus_prob)

        consensus_entropy = np.transpose(entropy(np.transpose(consensus_prob)))

        if self.debug:
            self.logger.log_math('Consensus Entropy:')
            print(consensus_entropy)

        learner_KL_divergence = np.zeros((ROLLING_WINDOW, NUM_LEARNERS))
        for i in range(ROLLING_WINDOW):
            for j in range(NUM_LEARNERS):
                learner_KL_divergence[i, j] = entropy(committee_merged_pred_window[j, i], qk=consensus_prob[i])

        if self.debug:
            self.logger.log_math('Kullback-Leibler Divergence:')
            print(learner_KL_divergence)

        max_disagreement = np.max(learner_KL_divergence, axis=1)

        if self.debug:
            self.logger.log_math('Max Disagreement:')
            print(max_disagreement)

        self.max_disagreement = max_disagreement

        return max_disagreement

    def evaluate_trigger_conditions(self, max_disagreement):
        query_decision = False

        samples_over_threshold = 0
        for i in range(ROLLING_WINDOW):
            if self.max_disagreement[i] > THRESHOLD_MAX_DISAGREEMENT_WINDOW:
                samples_over_threshold = samples_over_threshold + 1
        
        if self.max_disagreement[ROLLING_WINDOW - 1] > THRESHOLD_MAX_DISAGREEMENT_INDIVIDUAL:
            query_decision = True

        percent_over_threshold = samples_over_threshold / ROLLING_WINDOW
        if self.debug:
            print("Percent over threshold:", percent_over_threshold)

        if percent_over_threshold > THRESHOLD_PERCENT_OF_WINDOW:
            query_decision = True

        if query_decision:
            if self.time_since_last_query >= QUERY_LOCK_LENGTH:
                self.time_since_last_query = 0
            else:
                query_decision = False
                self.time_since_last_query = self.time_since_last_query + 1
        else:
            self.time_since_last_query = self.time_since_last_query + 1

        return query_decision

    # Class Methods

    def insert_sample(self, committee_vote_1, committee_vote_2, committee_vote_3, true):
        self.insert_to_buffers(committee_vote_1, committee_vote_2, committee_vote_3, true)
        max_disagreement = self.calculate_max_disagreement()
        query_decision = self.evaluate_trigger_conditions(max_disagreement)
        return max_disagreement[ROLLING_WINDOW - 1], query_decision