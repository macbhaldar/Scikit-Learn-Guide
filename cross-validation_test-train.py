from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target

print(housing.DESCR)

data.head()

# To simplify future visualization, 
# transform the prices from the 100 (k$) range to the thousand dollars (k\$) range.
target *= 100
target.head()

# To solve this regression task, we will use a decision tree regressor.
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(data, target)

from sklearn.metrics import mean_absolute_error

target_predicted = regressor.predict(data)
score = mean_absolute_error(target, target_predicted)
print(f"On average, our regressor makes an error of {score:.4f} k$")

# split dataset.

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)
    
regressor.fit(data_train, target_train)

target_predicted = regressor.predict(data_train)
score = mean_absolute_error(target_train, target_predicted)
print(f"The training error of our model is {score:.4f} k$")


target_predicted = regressor.predict(data_test)
score = mean_absolute_error(target_test, target_predicted)
print(f"The testing error of our model is {score:.4f} k$")


from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error")
    
import pandas as pd

cv_results = pd.DataFrame(cv_results)
cv_results.head()

cv_results["test_error"] = -cv_results["test_score"]
cv_results.head(10)

import matplotlib.pyplot as plt

cv_results["test_error"].plot.hist(bins=10, edgecolor="black")
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Test error distribution")

print(f"The mean cross-validated testing error is: "
      f"{cv_results['test_error'].mean():.2f} k$")
      
print(f"The standard deviation of the testing error is: "
      f"{cv_results['test_error'].std():.2f} k$")

target.plot.hist(bins=20, edgecolor="black")
plt.xlabel("Median House Value (k$)")
_ = plt.title("Target distribution")

print(f"The standard deviation of the target is: {target.std():.2f} k$")

cv_results = cross_validate(regressor, data, target, return_estimator=True)
cv_results

cv_results["estimator"]

from sklearn.model_selection import cross_val_score

scores = cross_val_score(regressor, data, target)
scores
