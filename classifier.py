# -*- coding: utf-8 -*-
import os
import json
from sklearn.svm import LinearSVC
from process_email import ProcessEmail
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

class Classifier():

    def __init__(self, start_train=0, end_train=3000):

        self.email_labels = "data/SPAMTrain.label"
        self.p = ProcessEmail(start_train=start_train, end_train=end_train)
        self.p.get_train_test_set()
        self.lista_emails = self.p.train_set
        self.word_list = {}
        self.load_word_list()
        self.load_labels()

    def load_word_list(self):
        try:
            with open('word_list.json', 'r') as f:
                self.word_list = json.load(f)
        except FileNotFoundError:
            self.p.construct_word_list()
            self.word_list = self.p.word_list

    def load_labels(self):

        with open(self.email_labels, 'r') as f:
            file_name_label = f.read().split()

            self.labels = list(zip(*[iter(reversed(file_name_label))]*2))
            self.labels = dict(self.labels)

    def train(self):

        self.model = LinearSVC(max_iter=10000)
        self.matrix_input, self.matrix_output = self.get_matrix_input_output()
        self.model.fit(self.matrix_input, self.matrix_output)

    def get_matrix_input_output(self):

        matrix_input = []
        matrix_output = []

        for email, email_file_name in self.lista_emails:

            label, word_indices = self._get_input_array(email, email_file_name)
            matrix_input.append(word_indices)
            matrix_output.append(label)

        return (np.asmatrix(matrix_input), np.asarray(matrix_output))

    def _get_input_array(self, email, email_file_name):
        label = self.labels[email_file_name]
        input_array = []
        for word in self.word_list:
            value = 0
            if word in email.split():
                value = 1

            input_array.append(value)
        return label, input_array

    def classify(self):

        lista_emails = self.p.test_set

        self.matrix_input_teste = []
        self.matrix_output_teste = []
        for tupla in lista_emails:

            email = tupla[0]
            email_file_name = tupla[1]
            label, word_indices = self._get_input_array(email, email_file_name)
            self.matrix_input_teste.append(word_indices)
            self.matrix_output_teste.append([label])

        output = self.model.predict(np.asmatrix(self.matrix_input_teste))
        true_output = np.asmatrix(self.matrix_output_teste)
        print("Accuracy %f" % f1_score(output, true_output, pos_label='1'))
        print("Confusion matrix")
        cm = confusion_matrix(output, true_output)
        print(cm)
        # import pylab as plt
        # labels = ["SPAM", "HAM"]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # plt.title('Confusion matrix of the classifier')
        # fig.colorbar(cax)
        # ax.set_xticklabels([''] + labels)
        # ax.set_yticklabels([''] + labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()