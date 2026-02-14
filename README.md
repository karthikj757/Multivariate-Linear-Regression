# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br>

### Step2
<br>

### Step3
<br>

### Step4
<br>

### Step5
<br>

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

# ✅ load the California housing dataset (replacement for Boston)
housing = datasets.fetch_california_housing()

# defining feature matrix (X) and response vector (y)
X = housing.data
y = housing.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print("Coefficients:", reg.coef_)

# variance score: 1 means perfect prediction
print("Variance score: {}".format(reg.score(X_test, y_test)))

# plot for residual error
plt.style.use("fivethirtyeight")

# plotting residual errors in training data
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data",
)

# plotting residual errors in test data
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data",
)

# plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=max(reg.predict(X_test)), linewidth=2)

# plotting legend
plt.legend(loc="upper right")

# plot title
plt.title("Residual errors")

# show plot
plt.show()

```
## Output:

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
