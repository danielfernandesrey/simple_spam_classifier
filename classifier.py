# -*- coding: utf-8 -*-
import os
import json
from sklearn.svm import LinearSVC
from process_email import ProcessEmail
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.externals import joblib

class Classifier():

    testing_emails = None
    svm_model = None
    start_training = None
    end_training = None

    def __init__(self, start_training=0, end_training=3000):
        self.start_training = start_training
        self.end_training = end_training

        self.p = ProcessEmail()
        self.word_list = self.p.load_word_list()

        self.load_input_outout_matrix()

        self.labels = self.p.load_labels()

        self.X_training = self.matrix_training_input[self.start_training:self.end_training]
        self.y_training = self.matrix_training_output[self.start_training:self.end_training]
        self.X_test = self.matrix_training_input[self.end_training:]
        self.y_test = self.matrix_training_output[self.end_training:]


    def load_input_outout_matrix(self):
        """
        Load the input and output matrix.
        Since the training matrix takes to long to generate, this method tries to load a alread trained matrix from
        a file.
        """

        try:

            self.matrix_training_input = joblib.load("matrix_input")
            self.matrix_training_output = joblib.load("matrix_output")

        except FileNotFoundError:

            training_emails = self.p.load_training_emails()

            matrix_input = []
            matrix_output = []

            for email_file_name, email in training_emails.items():

                label, word_indices = self._get_input_array(email, email_file_name)
                matrix_input.append(word_indices)
                matrix_output.append(label)

            self.matrix_training_input = np.asmatrix(matrix_input)
            self.matrix_training_output = np.asarray(matrix_output)

    def _get_input_array(self, email, email_file_name):
        """
        Create a 0 and 1 array. The array is created based on the words that appear both in the word list and the email
        content.
        :param email: Content of the email.
        :param email_file_name: Email file name.
        :return: 1 - For ham and 0 - For spam and the array of 0's and 1's
        """
        try:
            label = self.labels[email_file_name]
        except Exception:
            label = email_file_name
        input_array = []
        for word in self.word_list:
            value = 0
            if word in email.split():
                value = 1

            input_array.append(value)
        return label, input_array

    def run_svm(self):
        """
        Run the svm algorithm and output i'ts accuracy and the confusion matrix.
        """

        self.svm_model = LinearSVC(max_iter=20000)
        self.svm_model.fit(self.X_training, self.y_training)

        output_predictions = self.svm_model.predict(self.X_test)
        ground_truth_output = self.y_test

        score = f1_score(output_predictions, ground_truth_output, pos_label='1')
        print("Accuracy SVM %.2f%%" % (score*100))
        print("Confusion matrix")
        cm = confusion_matrix(output_predictions, ground_truth_output)
        print(cm)

    def run_neural_net(self):
        """
        Build a Neural Net using Keras, classify the emails with it and ouput i'ts accuracy.
        """

        from keras.models import Sequential
        from keras.layers import Dense
        self.neural_net = Sequential()
        self.neural_net.add(Dense(12, input_dim=self.X_training.shape[1], activation='relu'))
        self.neural_net.add(Dense(8, activation='relu'))
        self.neural_net.add(Dense(1, activation='sigmoid'))

        self.neural_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.neural_net.fit(self.X_training, self.y_training, epochs=300, batch_size=100, verbose=0)
        scores = self.neural_net.evaluate(self.X_training, self.y_training)
        print("\nAccuracy Neural Net: %.2f%%" % (scores[1]*100))
