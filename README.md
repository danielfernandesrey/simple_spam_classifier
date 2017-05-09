This is a work in progress. I'm new in the field of Machine Learning and I'm implementing some projects so I can lear.<br />
This project currently uses SVM to classify an e-mail as SPAM (0) or HAM (1).<br />

To run the classifier, just follow these steps:<br />
    - classifier = Classifier()<br />
    - classifier.train()<br />
    - classifier.classify()<br />

The algorithms trains the classifier with 3000 e-mails and, then, classify the rest of the emails of the dataset.
If a start and end values are passed to the classify method, then the classification is done only with the e-mails
in the range defined.