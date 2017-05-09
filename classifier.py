# -*- coding: utf-8 -*-
import os
import json
from sklearn.svm import LinearSVC
from process_email import ProcessEmail
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

class Classifier():

    def __init__(self):
        self.email_labels = "data/SPAMTrain.label"
        self.p = ProcessEmail()
        self.lista_emails = self.p.load_emails("data/training")
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

            self.labels = list(zip((*[iter(reversed(file_name_label))]*2)))
            self.labels = dict(self.labels)

    def get_model(self):

        self.model = LinearSVC()
        self.matriz_input, self.matriz_output = self.create_word_indices()
        self.model.fit(self.matriz_input, self.matriz_output)

    def create_word_indices(self):

        import numpy

        matriz_input = []
        matriz_output = []

        for email, email_file_name in self.lista_emails:

            label, word_indices = self._create_word_indices(email, email_file_name)

            matriz_input.append(word_indices)
            matriz_output.append(label)

        return (numpy.asmatrix(matriz_input), numpy.asarray(matriz_output))

    def _create_word_indices(self, email, email_file_name):
        label = self.labels[email_file_name]
        word_indices = []
        for word in self.word_list:
            value = 0
            if word in email.split():
                value = 1

            word_indices.append(value)
        return label, word_indices

    def classify(self, start=None, end=None):
        self.get_model()
        lista_emails = self.p.load_emails("data/training", start, end)
        scores = []

        self.matriz_input_teste = []
        self.matriz_output_teste = []
        for tupla in lista_emails:

            email = tupla[0]
            email_file_name = tupla[1]
            label, word_indices = self._create_word_indices(email, email_file_name)
            self.matriz_input_teste.append(word_indices)
            self.matriz_output_teste.append([label])

        output = self.model.predict(np.asmatrix(self.matriz_input_teste))
        true_output = np.asmatrix(self.matriz_output_teste)
        print("Score %f" % f1_score(output, true_output, pos_label='1'))
        print("Confusion matrix")
        print(confusion_matrix(output, true_output))
        #self.model.predict()












            # if predicted_label[0] != label:
            #
            #     print("Label original:", label)
            #     print("Predict label:", predicted_label)
            #     print("\n------------------------\n")
