import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from colorama import Fore, Style
from time import perf_counter, sleep, strftime
import csv

from CASAS_committee_predict import CASASCommitteePredict
from query_select import QuerySelect
from log import Log

class QueryProcessControl(object):
    def __init__(self):
        self.id = 'query_process_control'

        self.logger = Log(self.id)
        self.logger.startup_msg()

        # set to True for real-time mode (1 second/sample)
        self.real_time = True

        self.max_predictions = 0

        self.committee_predict = CASASCommitteePredict()
        self.query_select = QuerySelect()

        self.create_csv()

        self.logger.log_great('Ready.')

    def run(self):
        self.committee_predict.reset_counter()
        self.max_predictions = self.committee_predict.get_max_predictions()

        for i in range(0, self.max_predictions):
            if i > 0 and self.real_time:
                start_time = perf_counter()

            committee_vote_1, committee_vote_2, committee_vote_3, true = self.committee_predict.next_prediction()
            max_disagreement, query_decision = self.query_select.insert_sample(committee_vote_1, committee_vote_2, committee_vote_3, true)
            self.csv_log(committee_vote_1, committee_vote_2, committee_vote_3, true, max_disagreement, query_decision)

            if i > 0 and self.real_time:
                end_time = perf_counter()
                time_taken = end_time - start_time
                delay_time = 1.0 - time_taken

                if delay_time >= 0.0:
                    print('Prediction time was:', time_taken, ', sleeping for:', delay_time, 'seconds')
                    sleep(delay_time)
                else:
                    self.logger.log_warn('Predict/analyse cycle took longer than 1 second! System is not keeping up with real-time.')

    # Logging

    def create_csv(self):
        date_time = strftime("%Y%m%d-%H%M%S")
        self.csv_filename = 'logs/output_' + date_time + '.csv'

        msg = 'The logfile for this session is: ' + self.csv_filename
        self.logger.log(msg)

        with open(self.csv_filename, 'w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(["Learner 1", "Learner 2", "Learner 3", "Truth", "Max Disagreement", "Query Decision"])

    def csv_log(self, committee_vote_1, committee_vote_2, committee_vote_3, true, max_disagreement, query_decision):
        committee_vote_1, committee_vote_2, committee_vote_3, true = self.inverse_transform_labels(committee_vote_1, committee_vote_2, committee_vote_3, true)
        log_row = [committee_vote_1, committee_vote_2, committee_vote_3, true, max_disagreement, query_decision]
        with open(self.csv_filename, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(log_row)

    # Utilities

    def inverse_transform_labels(self, committee_vote_1, committee_vote_2, committee_vote_3, true):
        committee_vote_1 = np.argmax(committee_vote_1)
        committee_vote_2 = np.argmax(committee_vote_2)
        committee_vote_3 = np.argmax(committee_vote_3)

        committee_vote_1 = self.committee_predict.get_label(committee_vote_1)
        committee_vote_2 = self.committee_predict.get_label(committee_vote_2)
        committee_vote_3 = self.committee_predict.get_label(committee_vote_3)
        true = self.committee_predict.get_label(true)

        return committee_vote_1, committee_vote_2, committee_vote_3, true

if __name__ == '__main__':
    qpc = QueryProcessControl()
    qpc.run()