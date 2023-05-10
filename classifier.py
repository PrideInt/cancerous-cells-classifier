from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as panda
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()

# Because the data is basically everywhere and unreadable/unanalyzable, we will use pandas to generate some
# sort of visual in the form of a table/data frame

np_data = np.c_[data['data'], data['target']]
df_columns = np.append(data.feature_names, ['target'])

data_frame = panda.DataFrame(np_data, columns=df_columns)
table = data_frame.head()
print(table)

# We'll use a heatmap to take a better look at our data correlation
sns.heatmap(data_frame.corr(), annot=True, cmap='coolwarm')
plt.show()

# Visualization of our data
# We will visualize a handful of our feature data so everything does not look so fragmented
sns.pairplot(data_frame, palette='inferno', hue='target', vars=['mean smoothness', 'mean radius', 'mean area'])
plt.show()

# Feel free to look and compare with all possible features
# mean radius, mean texture, mean perimeter, mean area, mean smoothness
# mean compactness, mean concavity, mean concave points, mean symmetry
# mean fractal dimension, radius error, texture error, perimeter error
# area error, smoothness error, compactness error, concavity error
# concave points error, symmetry error, fractal dimension error
# worst radius, worst texture, worst perimeter, worst area
# worst smoothness, worst compactness, worst concavity
# worst concave points, worst symmetry, worst fractal dimension

# Get trained/test data with a test size of 0.5
x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.5, random_state=42)

# Visualize our trained data
sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], palette='inferno', hue=y_train, s=10)
plt.show()

def visualize_confusion_matrix(xt, yt, model):
    sns.heatmap(confusion_matrix(yt, model.predict(xt)), fmt='d', cmap='RdBu', square=True, annot=True, xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.show()

def run_classifiers(x=x_train, y=y_train, xt=x_test, yt=y_test):
    # Use Naive Bayes
    model = GaussianNB()

    # Train our data using Naive Bayes
    model.fit(x, y)

    # Generate predictions
    print(confusion_matrix(yt, model.predict(xt)))
    print(accuracy_score(yt, model.predict(xt)))
    print()

    visualize_confusion_matrix(xt, yt, model)

    # Use Support Vector Machines
    svm = SVC(kernel='linear')

    # Train our data using SVMs
    svm.fit(x, y)

    # Generate predictions
    print(confusion_matrix(yt, svm.predict(xt)))
    print(accuracy_score(yt, svm.predict(xt)))
    print()

    visualize_confusion_matrix(xt, yt, svm)

    # Use Logistic Regression
    log_reg = LogisticRegression()

    # Train our data using Logistic Regression
    log_reg.fit(x, y)

    # Generate predictions
    print(confusion_matrix(yt, log_reg.predict(xt)))
    print(accuracy_score(yt, log_reg.predict(xt)))
    print()

    visualize_confusion_matrix(xt, yt, log_reg)

# Now we run our classifiers on our trained data
run_classifiers()

# Now that we have our controls down, let's shuffle our data and see what we get
np.random.shuffle(x_train)
np.random.shuffle(x_test)
np.random.shuffle(y_train)
np.random.shuffle(y_test)

sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], palette='inferno', hue=y_train, s=10)
plt.show()

run_classifiers()