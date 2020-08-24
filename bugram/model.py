import heapq
import os
import pickle
from operator import itemgetter
from random import random
from typing import List

from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

from config import Config


class Model:

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        if self.config.LOAD_PATH:
            self.load_model(self.config.LOAD_PATH)
        print('Initialized model')

    def fit(self, sequences: List[List]):
        train, vocab = padded_everygram_pipeline(self.config.GRAM_SIZE, sequences)
        model = MLE(self.config.GRAM_SIZE)
        model.fit(train, vocab)
        self.model = model
        if self.config.SAVE_PATH:
            self.save_model(self.config.SAVE_PATH)

    def load_model(self, load_path):
        print("Loading model")
        with open(load_path, 'rb') as fin:
            self.model = pickle.load(fin)

    def save_model(self, save_path):
        print("Saving model")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as fout:
            pickle.dump(self.model, fout)

    def score(self, x, y, **kwargs):
        self.model.score(x, y, **kwargs)

    def evaluate(self, project_sequences):
        """ Print config.REPORTING_SIZE methods with lowest probability from the sequences """
        print("Starting to evaluate")
        shorter_sequence_list = self.get_low_probabilty_sequences(project_sequences, self.config.SHORTER_SEQUENCE_LEN)
        longer_sequence_list = self.get_low_probabilty_sequences(project_sequences, self.config.LONGER_SEQUENCE_LEN)
        print(type(shorter_sequence_list))
        union = self.get_union(shorter_sequence_list, longer_sequence_list)

        return union

    def get_union(self, shorter_sequence_list, longer_sequence_list):
        """ Find overlap between two sequences.
            For example: BCD in shorter list and ABCDE are reported as overlap

            This method can be inefficient since the config.REPORTING_SIZE is expected to be <= 100

            :param shorter_sequence_list: ('method_name', (probability, ['token1', 'token2', ..., 'token_n']))
            :param SHORTER_SEQUENCE_LEN: integer
            :param longer_sequence_list: ('method_name', (probability, ['token1', 'token2', ... , 'token_n']))
            :param LONGER_SEQUENCE_LEN: integer
            :return: Union in form ('method_name', (probability, ['token1', 'token2', ... , 'token_n'])) where tokens
            are taken from the longer sequence
        """
        union = []
        for short_method_sequence in shorter_sequence_list:
            short_sequence = short_method_sequence[1][1]
            for long_method_sequence in longer_sequence_list:
                long_sequence = long_method_sequence[1][1]
                intersection = [x for x in short_sequence if x in long_sequence]
                if len(intersection) == len(short_sequence):
                    if long_method_sequence not in union:
                        union.append(long_method_sequence)
        return union

    def get_low_probabilty_sequences(self, project_sequences, sequence_len):
        """ Given processed sequences in form "method_name token1,token2,token3"
            * This method will split those token sequences into fixed lengths
            * Then it will get probability for each of those sequences and
              returns the n lowest ones specified in self.config.REPORTING_SIZE
        """
        scoring_dict = {}
        for line in project_sequences:
            method_name = line.split(self.config.METHOD_TOKEN_SPLITTER)[self.config.METHOD_INDEX]
            tokens = line.split(self.config.METHOD_TOKEN_SPLITTER)[self.config.TOKEN_INDEX].split(
                self.config.TOKEN_SPLITTER)
            token_sequence_chunks = list(self.chunks(tokens, sequence_len))
            for chunk in token_sequence_chunks:
                score = self.score_sequence(chunk)
                scoring_dict[method_name + str(random())] = (score, chunk)
        return heapq.nsmallest(self.config.REPORTING_SIZE, scoring_dict.items(), key=itemgetter(1))

    def score_sequence(self, sequence):
        """ Scores a token sequence according to the following algorithm
            2-gram model example
                P(s) = P(w1)P(w2|w1)P(w3|w2)P(w4|w3)P(w5|w4)
            3-gram model example
                P(s) = P(w1)P(w2|w1)P(w3|w1w2)P(w4|w2w3)P(w5|w3w4)
            4-gram model example
                P(s) = P(w1)P(w2|w1)P(w3|w1w2)P(w4|w1w2w3)P(w5|w2w3w4)
            This extends to n-grams specified in config.py
        """
        # Break point to switch from increasing condition size to a fixed one
        break_point = self.config.GRAM_SIZE - 1
        total_score = 0
        for idx, token in enumerate(sequence):
            if idx < break_point:
                condition = sequence[:idx]
                temp_score = self.model.score(token, condition)
                if total_score is 0:
                    total_score = temp_score
                else:
                    total_score *= temp_score
            else:
                condition = sequence[idx - break_point:idx]
                temp_score = self.model.score(token, condition)
                if total_score is 0:
                    total_score = temp_score
                else:
                    total_score *= temp_score
        return total_score

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
