# -*- coding: utf-8 -*-

import re
import os
import json
from stemming.porter2 import stem


class ProcessEmail():
    word_list = {}

    def construct_word_list(self):
        """
        Construct a dict with the 100 hundred most common words and save the dict to a json file.
        """

        for processed_email, email_file in self.train_set:
            [self._create_word_list(word) for word in processed_email.split() if len(word) > 1]

        for key, value in list(self.word_list.items()):
            if value < 100:
                del self.word_list[key]

        with open("word_list.json", "w") as f:
            json.dump(self.word_list, f)

    def load_training_emails(self):
        """
        Start the preprocess of the emails.
        :return: 
        """
        email_dir = "data/training"
        training_emails = sorted(os.listdir(email_dir))
        return self._preprocess_set(training_emails, email_dir)

    def _preprocess_set(self, emails, email_dir):

        dicionario_emails = {}

        for email_file_name in emails:
            email_abspath = os.path.join(email_dir, email_file_name)
            with open(email_abspath, encoding='latin-1') as f:
                conteudo = f.read()
                dicionario_emails[email_file_name] = self._preprocess_email(conteudo)
        return dicionario_emails
                # yield (self.preprocess_email(conteudo), email)

    def _preprocess_email(self, email_content):
        """
        The following actions are performed to preprocess the emails:
            - Apply lowercase;
            - Stripping HTML;
            - Normalizing URLs;
            - Normalizing Email Addresses;
            - Normalizing Numbers;
            - Normalizing Dollars;
            - Word Stemming;
            - Removal of non-words;
        In the end, a word list is created.
        :param email_content:
        """

        email_content = email_content.lower()
        email_content = re.sub("<[^<>]+>", "", email_content)
        email_content = re.sub("[0-9]+", "number", email_content)
        email_content = re.sub("(http|https)://[^\s]*", "httpaddr", email_content)
        email_content = re.sub("[^\s]+@[^\s]+", "emailaddr", email_content)
        email_content = re.sub("[$]+", "dollar", email_content)
        email_content = " ".join([stem(word) for word in email_content.split()])
        email_content = re.sub('[^a-zA-Z]', ' ', email_content)
        email_content = re.sub("[\s]{2,}", " ", email_content)
        return email_content

    def _create_word_list(self, word):
        """
        Add only new words to a dict.
        :param word: (str)
        :return: 
        """
        if word not in self.word_list:
            self.word_list[word] = 1
        else:
            qtd = self.word_list[word]
            qtd = qtd + 1
            self.word_list[word] = qtd

    def load_test_set(self):

        dir_test = "data/testing"
        test_emails = os.listdir(dir_test)
        return self._preprocess_set(test_emails, dir_test)

    # p = ProcessEmail()
    # p.load_emails()

    def load_labels(self):

        email_labels = "data/SPAMTrain.label"
        with open(email_labels, 'r') as f:
            file_name_label = f.read().split()

            labels = list(zip(*[iter(reversed(file_name_label))] * 2))
            labels = dict(labels)
            return labels

    def load_word_list(self):
        try:
            with open('word_list.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.construct_word_list()
            return self.word_list
