# Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# Loading dataset
wine = pd.read_csv('../input/winequality-white.csv', sep=';')

# Let's check how the data is distributed
wine.head()

# Information about the data columns
wine.info()

fig = plt.figure(figsize=(10, 6))
corr = wine.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
fig.show()

# Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='fixed acidity', data=wine)
fig.show()

# Here we see that its quite a downing trend in the volatile acidity as we go higher the quality
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='volatile acidity', data=wine)
fig.show()

# Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='citric acid', data=wine)
fig.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='residual sugar', data=wine)
fig.show()

# Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='chlorides', data=wine)
fig.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='free sulfur dioxide', data=wine)
fig.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='total sulfur dioxide', data=wine)
fig.show()

# Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='sulphates', data=wine)
fig.show()

# Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='alcohol', data=wine)
fig.show()

# Preprocessing Data for performing Machine learning algorithms
# Making binary classificaion for the response variable.
# Dividing wine as good and bad by giving the limit for the quality
# quality (2,6] -> bad; (6, 9] -> good
bins = (2, 6, 9)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)

# Now lets assign a labels to our quality variable
label_quality = LabelEncoder()

# Bad becomes 0 (1) and good becomes 1 (2)
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()

fig = plt.figure(figsize=(10, 6))
sns.countplot(wine['quality'])
fig.show()

# Now separate the dataset as response variable and feature variabes
X = wine.drop('quality', axis=1)
y = wine['quality']

# Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Our training and testing data is ready now to perform machine learning algorithm

###
# Random Forest Classifier
# A random forest is a meta estimator that fits a number of decision tree classifiers on various
# sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# The sub-sample size is always the same as the original input sample size
# but the samples are drawn with replacement if bootstrap=True (default).
# n_estimators - The number of trees on the forest

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

# Let's see how our model performed
print(classification_report(y_test, pred_rfc))

# Random forest gives the accuracy of 88%

# Confusion matrix for the random forest classification
# The diagonal elements - the number of points for which the predicted label is equal to the true label
# off-diagonal elements - mislabeled by the classifier.
print(confusion_matrix(y_test, pred_rfc))

###
# Stochastic Gradient Descent Classifier
# The advantages of SGD:
# - Efficiency.
# - Ease of implementation (lots of opportunities for code tuning).
# The disadvantages of SGD:
# - SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
# - SGD is sensitive to feature scaling.

sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(sgd.coef_)

print(classification_report(y_test, pred_sgd))

# 77% accuracy using stochastic gradient descent classifier

print(confusion_matrix(y_test, pred_sgd))

###
# Support Vector Classifier
# fit time complexity is more than quadratic with the number of samples -> hard to scale more than 10000 samples.

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))

# 81% accuracy using support vector classifier

print(confusion_matrix(y_test, pred_svc))

###
# Let's try to increase our accuracy of models
# Grid Search CV

# Finding best parameters for our SVC model
# param = {
#     'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
#     'kernel': ['linear', 'rbf'],
#     'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
# }
# grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
#
# grid_svc.fit(X_train, y_train)
#
# # Best parameters for our svc model
# print(grid_svc.best_params_)
#
# # Let's run our SVC again with the best parameters.
# svc2 = SVC(C=1.2, gamma=0.9, kernel='rbf')
# svc2.fit(X_train, y_train)
# pred_svc2 = svc2.predict(X_test)
# print(classification_report(y_test, pred_svc2))
#
# # SVC improves from 81% to 85% using Grid Search CV
#
# # Cross Validation Score for random forest and SGD
#
# # Now lets try to do some evaluation for random forest model using cross validation.
# rfc_eval = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
# print rfc_eval.mean()
print('end')
