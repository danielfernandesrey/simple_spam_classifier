This is a work in progress. I'm new in the field of Machine Learning and I'm implementing some projects so I can lear.<br />
This project currently uses SVM to classify an e-mail as SPAM (0) or HAM (1).<br />

The classifier can use two algorithms to classify the data.
One is SVM and the other is a Neural Network. The Neural Net was created using Keras.


To run the classifier, just follow these steps:<br />
    - classifier = Classifier()<br />
    - classifier.run_svm() - To run the svm classifier<br />
    - classifier.classify() - To run the neural net classifier<br />

The algorithms trains the classifier with 3000 e-mails and, then, classify the rest of the emails of the dataset.
If a start and end values are feed to the classify method, then the algorithm classifies only the e-mails
in the range defined.