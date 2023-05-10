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
correctly identified to have benign tumors, and a 94.39% effectiveness in classification.

> 7 patients were incorrectly identified to have malignant tumors and 9 patients were incorrectly
> identified to have benign tumors.

### Logistic regression

![logreg confusion matrix.png](readme%2Flogreg%20confusion%20matrix.png)

![logreg effectiveness.png](readme%2Flogreg%20effectiveness.png)

We find that 90 patients were correctly identified to have a tumor that is malignant, 180 patients were 
correctly identified to have benign tumors, and a 94.74% effectiveness in classification.

> 7 patients were incorrectly identified to have malignant tumors and 8 patients were incorrectly
> identified to have benign tumors.

### Support vector machines

![svm confusion matrix.png](readme%2Fsvm%20confusion%20matrix.png)

![svm effectiveness.png](readme%2Fsvm%20effectiveness.png)

We find that 93 patients were correctly identified to have a tumor that is malignant, 182 patients were 
correctly identified to have benign tumors, and a 96.49% effectiveness in classification.

> 5 patients were incorrectly identified to have malignant tumors and 5 patients were incorrectly
> identified to have benign tumors.

### Conclusion

We can conclude that support vector machines are most accurate in predicting the cancerous nature of
a patient's tumor in this data set. Logistic regression falls behind support vector machines, however,
both are more accurate than naive Bayes predictions according to the confusion matrices.

## Scrambled data

To test the algorithms in a less controlled manner, I scrambled data to see the results. Surprisingly,
support vector machines and logistic regression algorithms provided more similar results to our previous
results as compared to naive Bayes.

Perhaps this is due to chance, however, in out of 10 trials, SVMs and logistic regression showed accuracy
greater than naive Bayes.

> ### Naive Bayes

![nb confusion matrix random.png](readme%2Fnb%20confusion%20matrix%20random.png)

> ### Logistic regression

![logreg confusion matrix random.png](readme%2Flogreg%20confusion%20matrix%20random.png)

> ### Support vector machines

![svm confusion matrix random.png](readme%2Fsvm%20confusion%20matrix%20random.png)

## Limitations

### No CTC cell count - metastasis

Unfortunately, this data set does not contain feature set data for
CTC (circulating tumor cell) count, which makes it impossible in this scenario
to predict the nature of metastasis within a patient.

Additionally, arbitrary sample number provided by the program would not be
effective or viable at all for this program.

### SVM decision boundary generation

This program unfortunately does not visualize the hyperplane nor the decision 
margins of the support vector machine, as computation and plotting of data 
was not accurate in visualizing this.

### Scrambling of untrained data

Due to the fact that much data within this data set are in fact, not 1s and 0s,
scrambling of untrained data could lead to many inaccuracies when fitting them
on any of the classifiers.

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