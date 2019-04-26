import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.neural_network import MLPRegressor
from math import sqrt


def data_to_plotly(x):
    k = []

    for i in range(0, len(x)):
        k.append(x.index[i])

    return k


# https://www.kaggle.com/raghavbirla/prediction-using-regression-and-classification

wine = pd.read_csv('../input/winequality-white.csv', ';')
wine.head()

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
target = ['quality']

# Checking for any null values in the dataset

X = wine[features]
y = wine[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=200)

###
# Linear regression
# Fit on train set
print('*' * 40)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print regressor.coef_

# Predict on test data
y_prediction = regressor.predict(X_test)
print(y_test[:5])
print(y_prediction[:5])
print y_test.describe()

fig = plt.figure(figsize=(10, 6))
p1 = plt.plot(x=data_to_plotly(X_train.alcohol), y=y_train,
              marker='o')
p2 = plt.plot(x=data_to_plotly(X_test.alcohol), y=y_prediction,
              marker='.')
fig.show()

# Evaluate Linear Regression accuracy using root-mean-square-error (RCSE) = 0,729
# In linear regression, the outcome (dependent variable) is continuous.
# It can have any one of an infinite number of possible values.
# In logistic regression, the outcome (dependent variable) has only a limited number of possible values.
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)


print '-------'
###
# Polynomial regression
# Fit on train set
model = PolynomialFeatures(degree=5)
y_ = model.fit_transform(y)
y_test_ = model.fit_transform(y_test)

lg = LinearRegression()
lg.fit(y_, X)
predicted_data = lg.predict(y_test_)
# predicted_data = np.round_(predicted_data)

print (mean_squared_error(X_test, predicted_data))
print lg.coef_
# print (predicted_data[:5])
print '-------'



###
# Decision Tree: Fit a new regression model to the training set
# max_depth - the maximum depth of the tree
print('*' * 40)
regressor = DecisionTreeRegressor(max_depth=50)
regressor.fit(X_train, y_train)
print regressor.class_weight

y_prediction = regressor.predict(X_test)
print y_test[:5]
print y_prediction[:5]
print y_test.describe()

# Evaluate Decision Tree Regression accuracy using root-mean-square-error (RCSE) = 0,813
# less RCSE -> better model (Linear regression is better)
RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)
