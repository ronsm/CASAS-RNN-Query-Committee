#!/usr/bin/env python3

import aiml
from log import Log
import responder

from responder import Responder
from semantic_similarity import SemanticSimilarity

class DialogueManager(object):
    def __init__(self, labels_dict):
        self.id = 'dialogue_manager'

        self.logger = Log(self.id)

        self.labels_dict = labels_dict

        self.logger.log_great('Ready.')

        self.aiml = aiml.Kernel()
        self.aiml.learn('std-startup.xml')
        self.aiml.respond('load aiml go')

        self.responder = Responder()
        self.semantic_similarity = SemanticSimilarity(self.labels_dict)
    
    def start_query(self, labels):
        for i in range(0, len(labels)):
            labels[i] = labels[i].lower()

        reduced, query_type = self.process_labels(labels)

        if query_type == 2:
            self.responder.query_2_labels(reduced)
        elif query_type == 3:
            self.responder.query_3_labels()
        else:
            self.logger.log_warn('A query process has been started when no query is required. Upstream error.')

        user_label = self.aiml.getPredicate('user_label')
        while user_label == '':
            self.get_input_and_respond()
            user_label = self.aiml.getPredicate('user_label')

        follow_up, options = self.semantic_similarity.compare_similarity(user_label, reduced)

        self.aiml.setPredicate('user_label', '')

        if follow_up:
            self.responder.query_2_labels_follow_up(options)

            user_label = self.aiml.getPredicate('user_label')
            while user_label == '':
                self.get_input_and_respond()
                user_label = self.aiml.getPredicate('user_label')

            follow_up, options = self.semantic_similarity.compare_similarity(user_label, options)

        self.responder.confirm_label(user_label)

    def process_labels(self, labels):
        reduced = []
        count = 0

        for label in labels:
            if label not in reduced:
                count = count + 1
                reduced.append(label)

        return reduced, count

    def get_input_and_respond(self):
        input = self.get_input()

        self.aiml.respond(input)
        method = self.aiml.getPredicate('responder')
        
        if method == 'bypass':
            return
        elif method != '':
            handle = getattr(self.responder, method)
            handle()
        else:
            self.logger.log_warn('No valid response.')
    
    def get_input(self):
        text = input('~>')
        return text