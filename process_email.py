# -*- coding: utf-8 -*-

import re
import os
import json
from stemming.porter2 import stem

class ProcessEmail():

    word_list = {}

    def __init__(self, email_content=None):
        self.email_dir = "data/training"

    def construct_word_list(self):
        """
        Construct a dict with the 100 hundred most common words and save the dict to a json file.
        """

        for processed_email, email_file in self.load_emails("data/training"):
            [self._create_word_list(word) for word in processed_email.split() if len(word) > 1]

        for key, value in list(self.word_list.items()):
            if value < 100:
                del self.word_list[key]

        with open("data.json", "w") as f:
            json.dump(self.word_list, f)

    def load_emails(self, email_dir=None, begin=0, end=4200):
        """
        Start the preprocess of the emails.
        :return: 
        """
        self.lista_emails_geral = sorted(os.listdir(email_dir))

        for email in self.lista_emails_geral[begin:end]:

            email_abspath = os.path.join(self.email_dir, email)
            with open(email_abspath, encoding='latin-1') as f:
                conteudo = f.read()
                yield (self.preprocess(conteudo),email)

    def preprocess(self, email_content):
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

#p = ProcessEmail()
#p.load_emails()