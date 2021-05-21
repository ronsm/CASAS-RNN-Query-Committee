import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from colorama import Fore, Style
from time import perf_counter, sleep, strftime
import csv
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from CASAS_committee_predict import CASASCommitteePredict
from query_select import QuerySelect
from dialogue_manager import DialogueManager
from CASAS_annotator import CASASAnnotator
from CASAS_AL_tools import CASASALTools
from label_linker import LabelLinker
from log import Log

QUERY_LIMIT = 4000
QUERY_RETRAIN = 25

sns.set_theme()

class QueryProcessControl(object):
    def __init__(self):
        self.id = 'query_process_control'

        self.logger = Log(self.id)
        self.logger.startup_msg()

        # * * * CONFIGURATION AREA * * * 

        # set to True for real-time mode (1 second/sample)
        self.real_time = False

        # set to True for automatic labelling
        self.oracle = True

        # set to True for automated re-training
        self.auto_al = True

        # set a limit on the number of predictions (for debug) - set to zero to disable
        self.max_predictions = 0

        # enable/disable debug mode (more verbose, particularly math operations) - caution: terminal lag
        self.debug = False

        # select the dataset
        self.dataset = "CASAS"

        # * * * * * * * * * * * * * * *

        self.sample_counter = 0
        self.num_queries = 0
        self.num_queries_at_last_retrain = 0

        self.learner_1_correct = 0
        self.learner_2_correct = 0
        self.learner_3_correct = 0

        self.learner_1_accuracies = []
        self.learner_2_accuracies = []
        self.learner_3_accuracies = []

        self.accuracy_query_markers = []
        self.samples_between_queries = 0

        if self.dataset == "CASAS":
            self.al_tools = CASASALTools()
            if self.auto_al:
                self.al_tools.init()
            self.committee_predict = CASASCommitteePredict(self.debug)
            self.annotator = CASASAnnotator(self.debug, self.dataset, self.committee_predict)
        else:
            self.logger.log_warn('Invalid dataset configuration.')

        self.query_select = QuerySelect(self.debug)
        self.label_linker = LabelLinker(self.dataset)
        self.dialogue_manager = DialogueManager(self.annotator, self.label_linker)

        self.create_csv()

        self.logger.log_great('Ready.')

    def run(self):
        self.committee_predict.reset_counter()

        if self.max_predictions == 0:
            self.max_predictions = self.committee_predict.get_max_predictions()

        for i in range(0, self.max_predictions):
            if i > 0 and self.real_time:
                start_time = perf_counter()

            committee_vote_1, committee_vote_2, committee_vote_3, true = self.committee_predict.next_prediction()
            current_sample = self.committee_predict.get_current_sample()
            self.annotator.add_sample(current_sample)
            max_disagreement, query_decision, disagreement_type = self.query_select.insert_sample(committee_vote_1, committee_vote_2, committee_vote_3, true)
            committee_vote_1, committee_vote_2, committee_vote_3, true = self.inverse_transform_labels(committee_vote_1, committee_vote_2, committee_vote_3, true)
            
            self.check_votes(committee_vote_1, committee_vote_2, committee_vote_3, true)

            self.csv_log(committee_vote_1, committee_vote_2, committee_vote_3, true, max_disagreement, query_decision)

            if query_decision:
                votes = [committee_vote_1, committee_vote_2, committee_vote_3]
                self.num_queries = self.num_queries + 1
                self.annotator.lock_buffer()
                if self.real_time:
                    threading.Thread(target=lambda: self.dialogue_manager.start_query(votes, true)).start()
                else:
                    if self.oracle:
                        self.annotator.annotate_buffer(true, true)
                    else:   
                        self.dialogue_manager.start_query(votes, true)

            # if query_decision:
            #     self.csv_log(committee_vote_1, committee_vote_2, committee_vote_3, true, disagreement_type, query_decision)

            if i > 0 and self.real_time:
                end_time = perf_counter()
                time_taken = end_time - start_time
                delay_time = 1.0 - time_taken

                if delay_time >= 0.0:
                    print('Prediction time was:', time_taken, ', sleeping for:', delay_time, 'seconds')
                    sleep(delay_time)
                else:
                    self.logger.log_warn('Predict/analyse cycle took longer than 1 second! System is not keeping up with real-time.')

            self.sample_counter = self.sample_counter + 1
            print('Progress:', self.sample_counter, 'of', self.max_predictions)

            if self.num_queries == QUERY_LIMIT:
                self.logger.log_warn('Query limit reached. Terminating.')
                break
            
            if self.auto_al:
                annotations_filename = self.annotator.get_annotation_filename()
                if (self.num_queries % QUERY_RETRAIN == 0) and (self.num_queries > 0) and (self.num_queries_at_last_retrain != self.num_queries):
                    self.al_tools.update(annotations_filename)
                    self.num_queries_at_last_retrain = self.num_queries

                    self.check_and_save_learner_accuracies()
                    self.reset_learner_correct_counts()
            else:
                if (self.num_queries % QUERY_RETRAIN == 0) and (self.num_queries > 0) and (self.num_queries_at_last_retrain != self.num_queries):
                    self.num_queries_at_last_retrain = self.num_queries

                    self.check_and_save_learner_accuracies()
                    self.reset_learner_correct_counts()

        # self.plot_learner_accuracies()
        self.plot_learner_val_accuracies()

    # Logging

    def create_csv(self):
        date_time = strftime("%Y%m%d-%H%M%S")
        self.csv_filename = 'logs/output_' + date_time + '.csv'

        msg = 'The logfile for this session is: ' + self.csv_filename
        self.logger.log(msg)

        with open(self.csv_filename, 'w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(["Sample Count", "Learner 1", "Learner 2", "Learner 3", "Truth", "Max Disagreement", "Query Decision"])

    def csv_log(self, committee_vote_1, committee_vote_2, committee_vote_3, true, disagreement_type, query_decision):
        # committee_vote_1, committee_vote_2, committee_vote_3, true = self.inverse_transform_labels(committee_vote_1, committee_vote_2, committee_vote_3, true)
        log_row = [self.sample_counter, committee_vote_1, committee_vote_2, committee_vote_3, true, disagreement_type, query_decision]
        with open(self.csv_filename, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(log_row)

    # Graphing

    def check_votes(self, committee_vote_1, committee_vote_2, committee_vote_3, true):
        print(committee_vote_1, committee_vote_2, committee_vote_3, true)

        if committee_vote_1 == true:
            self.learner_1_correct = self.learner_1_correct + 1
        
        if committee_vote_2 == true:
            self.learner_2_correct = self.learner_2_correct + 1
        
        if committee_vote_3 == true:
            self.learner_3_correct = self.learner_3_correct + 1

        self.samples_between_queries = self.samples_between_queries + 1

    def check_and_save_learner_accuracies(self):
        learner_1_accuracy = self.learner_1_correct / self.samples_between_queries
        learner_2_accuracy = self.learner_2_correct / self.samples_between_queries
        learner_3_accuracy = self.learner_3_correct / self.samples_between_queries

        self.learner_1_accuracies.append(learner_1_accuracy)
        self.learner_2_accuracies.append(learner_2_accuracy)
        self.learner_3_accuracies.append(learner_3_accuracy)

        self.accuracy_query_markers.append(self.num_queries)

        print(self.learner_1_accuracies, self.learner_2_accuracies, self.learner_3_accuracies)

    def reset_learner_correct_counts(self):
        self.samples_between_queries = 0

        self.learner_1_correct = 0
        self.learner_2_correct = 0
        self.learner_3_correct = 0

    def plot_learner_accuracies(self):
        plt.plot(self.accuracy_query_markers, self.learner_1_accuracies, marker='', color='blue', linewidth=2, label='Learner 1')
        plt.plot(self.accuracy_query_markers, self.learner_2_accuracies, marker='', color='red', linewidth=2, label='Learner 2')
        plt.plot(self.accuracy_query_markers, self.learner_3_accuracies, marker='', color='green', linewidth=2, label='Learner 3')

        plt.legend()

        plt.show()

    def plot_learner_val_accuracies(self):
        val_scores_learner_1, val_scores_learner_2, val_scores_learner_3 = self.al_tools.get_val_scores()

        self.accuracy_query_markers.insert(0, 0)

        plt.plot(self.accuracy_query_markers, val_scores_learner_1, marker='', color='blue', linewidth=2, label='Learner 1')
        plt.plot(self.accuracy_query_markers, val_scores_learner_2, marker='', color='red', linewidth=2, label='Learner 2')
        plt.plot(self.accuracy_query_markers, val_scores_learner_3, marker='', color='green', linewidth=2, label='Learner 3')

        plt.title('Learner Accuracy at Training Intervals of 25 Queries')
        plt.xlabel('No. Labels Gained from Queries')
        plt.ylabel('Accuracy (%) on Validation Set')

        plt.legend()

        plt.show()

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