#!/usr/bin/env python3

from log import Log
import spacy
import pprint
import numpy as np

nlp_eng = spacy.load('en_core_web_lg')

SIMILARITY_MARGIN = 0.2

class SemanticSimilarity(object):
    def __init__(self, labels_dict):
        self.id = 'semantic_similarity'

        self.logger = Log(self.id)

        self.labels_dict = labels_dict

    def compare_similarity(self, compare, labels):
        follow_up = False
        options = ['']

        if len(labels) == 2:
            similarity_scores = self.compute_similarity(compare, True, labels)
            similarity_scores_sorted = self.sort_similarity_scores(similarity_scores)
            return follow_up, options
        elif len(labels) == 3:
            similarity_scores = self.compute_similarity(compare)
            similarity_scores_sorted = self.sort_similarity_scores(similarity_scores)
            follow_up, options = self.evaluate_follow_up(similarity_scores_sorted)
            return follow_up, options
        else:
            self.logger.log_warn('A query process has been started when no query is required. Upstream error.')

    def compute_similarity(self, compare, reduced=False, labels=['']):
        all_similarity_scores = {}
        compare = nlp_eng(compare)
        
        for key, value in self.labels_dict.items():
            if not reduced:
                class_descriptions = []
                similarity_scores = []

                for item in value:
                    class_descriptions.append(nlp_eng(item))
                
                for class_description in class_descriptions:
                    similarity_score = class_description.similarity(compare)
                    similarity_scores.append(similarity_score)

                all_similarity_scores[key] = similarity_scores
            else:
                if key in labels:
                    class_descriptions = []
                    similarity_scores = []

                    for item in value:
                        class_descriptions.append(nlp_eng(item))
                    
                    for class_description in class_descriptions:
                        similarity_score = class_description.similarity(compare)
                        similarity_scores.append(similarity_score)

                    all_similarity_scores[key] = similarity_scores

        return all_similarity_scores

    def evaluate_follow_up(self, similarity_scores):
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

    def sort_similarity_scores(self, similarity_scores):
        similarity_scores_argmax = {}

        for key, value in similarity_scores.items():
            np_array = np.asarray(value)
            argmax = np_array[np_array.argmax()]

            similarity_scores_argmax[key] = argmax

        data_sorted = {k: v for k, v in sorted(similarity_scores_argmax.items(), reverse=True, key=lambda x: x[1])}

        print(data_sorted)

        return data_sorted