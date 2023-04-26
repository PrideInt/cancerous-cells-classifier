from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
data.target_names

from sklearn.model_selection import train_test_split

# Get trained/test data
train, test, training_labels, test_labels = train_test_split(data['data'], data['target'], test_size=0.8, random_state=114)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Implement Naive Bayes
model = GaussianNB()

# Train our data using Naive Bayes
model.fit(train, training_labels)

# Generate predictions
print(confusion_matrix(test_labels, model.predict(test)))
print(accuracy_score(test_labels, model.predict(test)))

# Use Support Vector Machines
svm = SVC()

# Train our data using SVMs
svm.fit(train, training_labels)

# Generate predictions
print(confusion_matrix(test_labels, svm.predict(test)))
print(accuracy_score(test_labels, svm.predict(test)))

# TBA