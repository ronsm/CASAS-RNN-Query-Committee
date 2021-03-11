#!/usr/bin/env python3

import aiml
from log import Log
import responder

from responder import Responder
from semantic_similarity import SemanticSimilarity

class DialogueManager(object):
    def __init__(self, labels_dict, annotator):
        self.id = 'dialogue_manager'

        self.logger = Log(self.id)

        self.labels_dict = labels_dict

        self.logger.log_great('Ready.')

        self.aiml = aiml.Kernel()
        self.aiml.learn('std-startup.xml')
        self.aiml.respond('load aiml go')

        self.responder = Responder()
        self.annotator = annotator
        self.semantic_similarity = SemanticSimilarity(self.labels_dict)
    
    def start_query(self, labels):
        for i in range(0, len(labels)):
            labels[i] = labels[i].lower()

        reduced, query_type = self.process_labels(labels)

        if query_type == 1:
            self.story_query_1_label(reduced[0])
        elif query_type == 2:
            self.story_query_2_labels(reduced)
        elif query_type == 3:
            self.story_query_3_labels(reduced)
        else:
            self.logger.log_warn('A query process has been started when no query is required. Upstream error.')

        self.annotator.unlock_buffer()

    # Stories

    def story_query_1_label(self, confirmation_label):
        self.responder.query_1_label(confirmation_label)

        affirm_label = self.aiml.getPredicate('affirm_label')
        while affirm_label == '':
            self.get_input_and_respond()
            affirm_label = self.aiml.getPredicate('affirm_label')

        if affirm_label == 'true':
            self.responder.confirm_label(confirmation_label)
            self.annotator.annotate_buffer(confirmation_label)
        elif affirm_label == 'false':
            user_label = self.aiml.getPredicate('user_label')
            if user_label != '':
                follow_up, options = self.semantic_similarity.compare_similarity(user_label, compare_all=True)
                if follow_up:
                    self.responder.query_2_labels_follow_up(options)

                    user_label = self.aiml.getPredicate('user_label')
                    while user_label == '':
                        self.get_input_and_respond()
                        user_label = self.aiml.getPredicate('user_label')

                    follow_up, options = self.semantic_similarity.compare_similarity(user_label, labels=options)

                self.responder.confirm_label(user_label)
                self.annotator.annotate_buffer(user_label)
            else:
                self.story_query_all_labels()
        else:
            self.logger.log_warn('Invalid affirmation response. AIML error.')

    def story_query_2_labels(self, reduced):
        self.responder.query_2_labels(reduced)

        user_label = self.aiml.getPredicate('user_label')
        while user_label == '':
            self.get_input_and_respond()
            user_label = self.aiml.getPredicate('user_label')

        follow_up, options = self.semantic_similarity.compare_similarity(user_label, labels=reduced)

        self.aiml.setPredicate('user_label', '')

        if follow_up:
            self.responder.query_2_labels_follow_up(options)

            user_label = self.aiml.getPredicate('user_label')
            while user_label == '':
                self.get_input_and_respond()
                user_label = self.aiml.getPredicate('user_label')

            follow_up, options = self.semantic_similarity.compare_similarity(user_label, labels=options)

        self.responder.confirm_label(user_label)
        self.annotator.annotate_buffer(user_label)

    def story_query_3_labels(self, reduced):
        self.responder.query_3_labels()

        user_label = self.aiml.getPredicate('user_label')
        while user_label == '':
            self.get_input_and_respond()
            user_label = self.aiml.getPredicate('user_label')

        follow_up, options = self.semantic_similarity.compare_similarity(user_label, labels=reduced)

        self.aiml.setPredicate('user_label', '')

        if follow_up:
            self.responder.query_2_labels_follow_up(options)

            user_label = self.aiml.getPredicate('user_label')
            while user_label == '':
                self.get_input_and_respond()
                user_label = self.aiml.getPredicate('user_label')

            follow_up, options = self.semantic_similarity.compare_similarity(user_label, labels=options)

        self.responder.confirm_label(user_label)
        self.annotator.annotate_buffer(user_label)

    def story_query_all_labels(self):
        self.responder.query_3_labels()

        user_label = self.aiml.getPredicate('user_label')
        while user_label == '':
            self.get_input_and_respond()
            user_label = self.aiml.getPredicate('user_label')

        follow_up, options = self.semantic_similarity.compare_similarity(user_label, compare_all=True)

        self.aiml.setPredicate('user_label', '')

        if follow_up:
            self.responder.query_2_labels_follow_up(options)

            user_label = self.aiml.getPredicate('user_label')
            while user_label == '':
                self.get_input_and_respond()
                user_label = self.aiml.getPredicate('user_label')

            follow_up, options = self.semantic_similarity.compare_similarity(user_label, labels=options)

        self.responder.confirm_label(user_label)
        self.annotator.annotate_buffer(user_label)

    # Tools

    def process_labels(self, labels):
        reduced = []
        count = 0

        for label in labels:
            if label not in reduced:
                count = count + 1
                reduced.append(label)

        semantic_descriptions = []
        for label in reduced:
            semantic_description = self.labels_dict.get(label)[0]
            semantic_descriptions.append(semantic_description)

        reduced = semantic_descriptions

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