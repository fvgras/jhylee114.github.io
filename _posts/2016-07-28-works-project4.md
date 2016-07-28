---
priority: 0.8
title: Project 4
excerpt: Web Scraping for Indeed.com & Predicting Salaries
categories: works
background-image: workspace.jpg
tags:
  - web scraping
  - logistic regression
  - cross-validation
  - gridsearchcv
---


# Web Scraping for Indeed.com & Predicting Salaries

In this project, we will practice two major skills: collecting data by scraping a website and then building a binary predictor with Logistic Regression.

We are going to collect salary information on data science jobs in a variety of markets. Then using the location, title and summary of the job we will attempt to predict the salary of the job. For job posting sites, this would be extraordinarily useful. While most listings DO NOT come with salary information (as you will see in this exercise), being to able extrapolate or predict the expected salaries from other listings can help guide negotiations.

Normally, we could use regression for this task; however, we will convert this problem into classification and use Logistic Regression.

- Question: Why would we want this to be a classification problem?
- Answer: While more precision may be better, there is a fair amount of natural variance in job salaries - predicting a range be may be useful.

Therefore, the first part of the assignment will be focused on scraping Indeed.com. In the second, we'll focus on using listings with salary information to build a model and predict additional salaries.

# Summary

## Predicting salaries using Logistic Regression

### Load the Data


```python
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/JHYL/DSI-HK-1/projects/project-04/starter-code/indeedscrape.csv')
df = pd.DataFrame(df, columns=['job_location', 'city', 'job_title', 'company', 'salary'])
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job_location</th>
      <th>city</th>
      <th>job_title</th>
      <th>company</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Data Scientist 5796057</td>
      <td>Avispa Technology</td>
      <td>100000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Research Scientist- Molecular &amp; Cellular Oncology</td>
      <td>MD ANDERSON CANCER CENTER</td>
      <td>64500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Interdisciplinary (Safety Engineer or Physical...</td>
      <td>Department Of The Interior</td>
      <td>113323.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Research Assistant I</td>
      <td>Baylor College of Medicine</td>
      <td>36000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Bioinformatics Programmer II</td>
      <td>Baylor College of Medicine</td>
      <td>70000.0</td>
    </tr>
  </tbody>
</table>
</div>



#### We want to predict a binary variable - whether the salary was low or high. Compute the median salary and create a new binary variable that is true when the salary is high (above the median)


```python
df['high_salary'] = df.salary >= np.median(df.salary)
df['high_salary'].head()
```




    0     True
    1    False
    2     True
    3    False
    4    False
    Name: high_salary, dtype: bool




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job_location</th>
      <th>city</th>
      <th>job_title</th>
      <th>company</th>
      <th>salary</th>
      <th>high_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Data Scientist 5796057</td>
      <td>Avispa Technology</td>
      <td>100000.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Research Scientist- Molecular &amp; Cellular Oncology</td>
      <td>MD ANDERSON CANCER CENTER</td>
      <td>64500.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Interdisciplinary (Safety Engineer or Physical...</td>
      <td>Department Of The Interior</td>
      <td>113323.5</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Research Assistant I</td>
      <td>Baylor College of Medicine</td>
      <td>36000.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Houston, TX</td>
      <td>Houston</td>
      <td>Bioinformatics Programmer II</td>
      <td>Baylor College of Medicine</td>
      <td>70000.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Binary Classification Problem
#### We could also perform Linear Regression (or any regression) to predict the salary value here. Instead, we are going to convert this into a _binary_ classification problem, by predicting two classes, HIGH vs LOW salary.

While performing regression may be better, performing classification may help remove some of the noise of the extreme salaries. We don't have to choice the `median` as the splitting point - we could also split on the 75th percentile or any other reasonable breaking point.

In fact, the ideal scenario may be to predict many levels of salaries, 

### Logistic Regression (Scikitlearn)
#### Rebuild this model with scikit-learn.
- You can either create the dummy features manually or use the `dmatrix` function from `patsy`
- Remember to scale the feature variables as well!



```python
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

X = pd.get_dummies(df['job_location'])
y = df['high_salary']

model = LogisticRegression()
model.fit(X, y)
print model

y_pred = model.predict(X)

print "Classification Report:"
print metrics.classification_report(y, y_pred)

print "Confusion Matrix:"
print metrics.confusion_matrix(y, y_pred)

print "Classification Scoring:"
print "Accuracy Score:", metrics.accuracy_score(y, y_pred)
print "AUC Score:", metrics.roc_auc_score(y, y_pred)
print "Precision Score:", metrics.precision_score(y, y_pred)
print "Recall Score:", metrics.recall_score(y, y_pred)
print "F1 Score:", metrics.f1_score(y, y_pred)

print "Regression Scoring:"
print "MAE:", metrics.mean_absolute_error(y, y_pred)
print "MSE:", metrics.mean_squared_error(y, y_pred)
print "R2:", metrics.r2_score(y, y_pred)
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.91      0.59      0.71       157
           True       0.78      0.96      0.86       246
    
    avg / total       0.83      0.82      0.81       403
    
    Confusion Matrix:
    [[ 92  65]
     [  9 237]]
    Classification Scoring:
    Accuracy Score: 0.816377171216
    AUC Score: 0.774700947646
    Precision Score: 0.784768211921
    Recall Score: 0.963414634146
    F1 Score: 0.86496350365
    Regression Scoring:
    MAE: 0.183622828784
    MSE: 0.183622828784
    R2: 0.227849412252


    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:164: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average(np.abs(y_pred - y_true),
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:232: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average((y_true - y_pred) ** 2, axis=0,
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:452: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,


### Cross Validation by Train Test Split (ScikitLearn)

#### Use cross-validation in scikit-learn to evaluate the model above. 
- Evaluate the accuracy, AUC, precision and recall of the model. 
- Discuss the differences and explain when you want a high-recall or a high-precision model in this scenario.


```python
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print "Classification Report:"
print classification_report(y_test, y_pred)

print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)

print "Classification Scoring:"
print "Accuracy Score:", metrics.accuracy_score(y_test, y_pred)
print "AUC Score:", metrics.roc_auc_score(y_test, y_pred)
print "Precision Score:", metrics.precision_score(y_test, y_pred)
print "Recall Score:", metrics.recall_score(y_test, y_pred)
print "F1 Score:", metrics.f1_score(y_test, y_pred)

print "Regression Scoring:"
print "MAE:", metrics.mean_absolute_error(y_test, y_pred)
print "MSE:", metrics.mean_squared_error(y_test, y_pred)
print "R2:", metrics.r2_score(y_test, y_pred)
```

    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.53      0.72      0.61        75
           True       0.79      0.63      0.70       127
    
    avg / total       0.70      0.66      0.67       202
    
    Confusion Matrix:
    [[54 21]
     [47 80]]
    Classification Scoring:
    Accuracy Score: 0.663366336634
    AUC Score: 0.674960629921
    Precision Score: 0.792079207921
    Recall Score: 0.629921259843
    F1 Score: 0.701754385965
    Regression Scoring:
    MAE: 0.336633663366
    MSE: 0.336633663366
    R2: -0.442099737533


    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:164: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average(np.abs(y_pred - y_true),
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:232: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average((y_true - y_pred) ** 2, axis=0,
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:452: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,


#### Thought experiment: What is the baseline accuracy for this model?


```python
print "Logistic Regression Accuracy Score: 82%"
print "Cross-Validation Accuracy Score: 69%"
```

    Logistic Regression Accuracy Score: 82%
    Cross-Validation Accuracy Score: 69%


#### Create a few new variables in your dataframe to represent interesting features of a job title.
- For example, create a feature that represents whether 'Senior' is in the title 
- or whether 'Manager' is in the title. 
- Then build a new Logistic Regression model with these features. Do they add any value? 


#### Logistic Regression with New Feature


```python
df['senior'] = df['job_title'].str.contains('Senior')
df['manager'] = df['job_title'].str.contains('Manager')
```


```python
X1 = df[['senior', 'manager']]
y = df['high_salary']

model1 = LogisticRegression()
model1.fit(X1, y)
print model1

y_pred1 = model1.predict(X1)

print "Classification Report:"
print metrics.classification_report(y, y_pred1)

print "Confusion Matrix:"
print metrics.confusion_matrix(y, y_pred1)

print "Classification Scoring:"
print "Accuracy Score:", metrics.accuracy_score(y, y_pred1)
print "AUC Score:", metrics.roc_auc_score(y, y_pred1)
print "Precision Score:", metrics.precision_score(y, y_pred1)
print "Recall Score:", metrics.recall_score(y, y_pred1)
print "F1 Score:", metrics.f1_score(y, y_pred1)

print "Regression Scoring:"
print "MAE:", metrics.mean_absolute_error(y, y_pred1)
print "MSE:", metrics.mean_squared_error(y, y_pred1)
print "R2:", metrics.r2_score(y, y_pred1)
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.58      0.04      0.08       157
           True       0.62      0.98      0.76       246
    
    avg / total       0.60      0.62      0.49       403
    
    Confusion Matrix:
    [[  7 150]
     [  5 241]]
    Classification Scoring:
    Accuracy Score: 0.615384615385
    AUC Score: 0.512130392005
    Precision Score: 0.616368286445
    Recall Score: 0.979674796748
    F1 Score: 0.756671899529
    Regression Scoring:
    MAE: 0.384615384615
    MSE: 0.384615384615
    R2: -0.61734244731


    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:164: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average(np.abs(y_pred - y_true),
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:232: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average((y_true - y_pred) ** 2, axis=0,
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:452: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,


#### Cross-Validation with New Feature


```python
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.50)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print "Classification Report:"
print classification_report(y_test, y_pred)

print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)

print "Classification Scoring:"
print "Accuracy Score:", metrics.accuracy_score(y_test, y_pred)
print "AUC Score:", metrics.roc_auc_score(y_test, y_pred)
print "Precision Score:", metrics.precision_score(y_test, y_pred)
print "Recall Score:", metrics.recall_score(y_test, y_pred)
print "F1 Score:", metrics.f1_score(y_test, y_pred)

print "Regression Scoring:"
print "MAE:", metrics.mean_absolute_error(y_test, y_pred)
print "MSE:", metrics.mean_squared_error(y_test, y_pred)
print "R2:", metrics.r2_score(y_test, y_pred)
```

    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.60      0.04      0.08        71
           True       0.65      0.98      0.79       131
    
    avg / total       0.64      0.65      0.54       202
    
    Confusion Matrix:
    [[  3  68]
     [  2 129]]
    Classification Scoring:
    Accuracy Score: 0.653465346535
    AUC Score: 0.513493172777
    Precision Score: 0.654822335025
    Recall Score: 0.984732824427
    F1 Score: 0.786585365854
    Regression Scoring:
    MAE: 0.346534653465
    MSE: 0.346534653465
    R2: -0.520266637996


    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:164: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average(np.abs(y_pred - y_true),
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:232: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average((y_true - y_pred) ** 2, axis=0,
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:452: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,


## Compare L1 and L2 regularization for this logistic regression model. What effect does this have on the coefficients learned?

#### L1 and L2 Regularization on Job Location - Salary Model


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
lr = LogisticRegression(penalty='l1')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print "Classification Report:"
print classification_report(y_test, y_pred)

print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)
```

    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.52      0.76      0.62        78
           True       0.79      0.56      0.66       124
    
    avg / total       0.68      0.64      0.64       202
    
    Confusion Matrix:
    [[59 19]
     [54 70]]



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
lr = LogisticRegression(penalty='l2')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print "Classification Report:"
print classification_report(y_test, y_pred)

print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)
```

    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.73      0.28      0.40        79
           True       0.67      0.93      0.78       123
    
    avg / total       0.69      0.68      0.63       202
    
    Confusion Matrix:
    [[ 22  57]
     [  8 115]]


#### L1 and L2 Regularization on Job Title - Salary Model


```python
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.50)
lr = LogisticRegression(penalty='l1')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print "Classification Report:"
print classification_report(y_test, y_pred)

print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)
```

    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.00      0.00      0.00        87
           True       0.57      1.00      0.73       115
    
    avg / total       0.32      0.57      0.41       202
    
    Confusion Matrix:
    [[  0  87]
     [  0 115]]


    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.50)
lr = LogisticRegression(penalty='l2')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print "Classification Report:"
print classification_report(y_test, y_pred)

print "Confusion Matrix:"
print confusion_matrix(y_test, y_pred)
```

    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.57      0.05      0.09        81
           True       0.61      0.98      0.75       121
    
    avg / total       0.59      0.60      0.48       202
    
    Confusion Matrix:
    [[  4  77]
     [  3 118]]


## GridSearchCV

#### Grid Search CV on Job Location - Salary Model


```python
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(LogisticRegression(class_weight='balanced'),
                   {"C":[0.01,0.1,1.0,10.0,100.0]},
                   n_jobs=-1)
clf.fit(X, y)

print "Best Estimator:"
print clf.best_estimator_

print "Best Parameters:"
print clf.best_params_

print "Best Score:"
print clf.best_score_
```

    Best Estimator:
    LogisticRegression(C=100.0, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
    Best Parameters:
    {'C': 100.0}
    Best Score:
    0.570719602978


#### Grid Search CV on Job Title - Salary Model


```python
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(LogisticRegression(class_weight='balanced'),
                   {"C":[0.01,0.1,1.0,10.0,100.0]},
                   n_jobs=-1)
clf.fit(X1, y)

print "Best Estimator:"
print clf.best_estimator_

print "Best Parameters:"
print clf.best_params_
print "Best Score:"
print clf.best_score_
```

    Best Estimator:
    LogisticRegression(C=0.01, class_weight='balanced', dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
    Best Parameters:
    {'C': 0.01}
    Best Score:
    0.578163771712


#### Continue to incorporate other text features from the title or summary that you believe will predict the salary and examine their coefficients


```python
df['director'] = df['job_title'].str.contains('Director')

X2 = df[['senior', 'manager', 'director']]
y = df['high_salary']

model1 = LogisticRegression()
model1.fit(X2, y)
print model1

y_pred2 = model1.predict(X2)

print "Classification Report:"
print metrics.classification_report(y, y_pred2)

print "Confusion Matrix:"
print metrics.confusion_matrix(y, y_pred2)

print "Classification Scoring:"
print "Accuracy Score:", metrics.accuracy_score(y, y_pred2)
print "AUC Score:", metrics.roc_auc_score(y, y_pred2)
print "Precision Score:", metrics.precision_score(y, y_pred2)
print "Recall Score:", metrics.recall_score(y, y_pred2)
print "F1 Score:", metrics.f1_score(y, y_pred2)


print "Regression Scoring:"
print "MAE:", metrics.mean_absolute_error(y, y_pred2)
print "MSE:", metrics.mean_squared_error(y, y_pred2)
print "R2:", metrics.r2_score(y, y_pred2)
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    Classification Report:
                 precision    recall  f1-score   support
    
          False       0.58      0.04      0.08       157
           True       0.62      0.98      0.76       246
    
    avg / total       0.60      0.62      0.49       403
    
    Confusion Matrix:
    [[  7 150]
     [  5 241]]
    Classification Scoring:
    Accuracy Score: 0.615384615385
    AUC Score: 0.512130392005
    Precision Score: 0.616368286445
    Recall Score: 0.979674796748
    F1 Score: 0.756671899529
    Regression Scoring:
    MAE: 0.384615384615
    MSE: 0.384615384615
    R2: -0.61734244731


    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:164: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average(np.abs(y_pred - y_true),
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:232: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      output_errors = np.average((y_true - y_pred) ** 2, axis=0,
    /Users/JHYL/anaconda/lib/python2.7/site-packages/sklearn/metrics/regression.py:452: DeprecationWarning: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
      numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,

