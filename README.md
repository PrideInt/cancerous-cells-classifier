# Classification of Cancerous Cells and Cancerous Nature of Tumors of Wisconsin Patients with Breast Cancer
Running naive Bayes and support vector machine classifiers using scikit-learn on benign/malignant cancer 
cell data sets to predict cancerous nature of the patient. Data accessed from the
**Breast Cancer Wisconsin (Diagnostic) Data Set**:
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic).
- **Pride Y**.

## Training data

We will train the data from this data set running three different classifier algorithms (**Naive Bayes**, 
**Logistic Regression** and **Support Vector Machines**).

### Data correlation

We find that some *features* in this data set are highly correlated, such as worst perimeter and worst
radius, and some little correlation, mean symmetry and mean smoothness.

![heatmap correlation.png](readme%2Fheatmap%20correlation.png)

### Visualization

We can also visualize the separability of our trained data.

![trained data scatter plot.png](readme%2Ftrained%20data%20scatter%20plot.png)

## Classifier

This program rudimentally classifies cancer cell data on benignity and malignancy from scikit-learn's 
existing breast cancer data set running **Naive Bayes**, **Logistic Regression** and 
**Support Vector Machine** classifiers.

We will train the data, then run it through a confusion matrix to identify the correctness of our data
for all three classifiers.

### Naive bayes

![nb confusion matrix.png](readme%2Fnb%20confusion%20matrix.png)

![nb effectiveness.png](readme%2Fnb%20effectiveness.png)

We find that 89 patients were correctly identified to have a tumor that is malignant, 180 patients were 
correctly identified to have benign tumors, and a 94.79% effectiveness in classification.

### Logistic regression

![logreg confusion matrix.png](readme%2Flogreg%20confusion%20matrix.png)

![logreg effectiveness.png](readme%2Flogreg%20effectiveness.png)

We find that 90 patients were correctly identified to have a tumor that is malignant, 180 patients were 
correctly identified to have benign tumors, and a 94.74% effectiveness in classification.

### Support vector machines

![svm confusion matrix.png](readme%2Fsvm%20confusion%20matrix.png)

![svm effectiveness.png](readme%2Fsvm%20effectiveness.png)

We find that 93 patients were correctly identified to have a tumor that is malignant, 182 patients were 
correctly identified to have benign tumors, and a 96.49% effectiveness in classification.

### Conclusion

We can conclude that support vector machines are most accurate in predicting the cancerous nature of
a patient's tumor in this data set.

## Scrambled data

To test the algorithms in a less controlled manner, I scrambled data to see the results. Surprisingly,
support vector machines and logistic regression algorithms provided more similar results as compared to
naive bayes.

> ### Naive Bayes

![nb confusion matrix random.png](readme%2Fnb%20confusion%20matrix%20random.png)

> ### Logistic regression

![logreg confusion matrix random.png](readme%2Flogreg%20confusion%20matrix%20random.png)

> ### Support vector machines

![svm confusion matrix random.png](readme%2Fsvm%20confusion%20matrix%20random.png)

## Dependencies

### scikit-learn
```js
pip install scikit-learn
```

### matplotlib
```js
pip install matplotlib
```

### pandas
```js
pip install pandas
```

### numpy
```js
pip install numpy
```

### seaborn
```js
pip install seaborn
```