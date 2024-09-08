# Movie-box-office-prediction

The revenue of a movie is influenced by various factors such as the actors, production costs, film critics' opinions, the movie's rating, the release year, and more. Given this complexity, there is no single formula that guarantees an accurate revenue forecast for a movie.

However, we can develop a model that analyzes these influencing factors to predict the potential revenue of a movie. This model helps in making informed predictions about the expected financial performance of a movie.

 I evaluated various regression techniques, including Linear Regression, Polynomial Regression, Ridge Regression, Logistic Regression, K-Nearest Neighbors Regression, Decision Tree, XGB Regressor, and Random Forest Regressor. The final analysis showed that the Random Forest Regressor delivered the most accurate result.


# Models Tried:

Logistic Regression : logistic regression (or logit regression) is estimating the parameters
of a logistic model (the coefficients in the linear combination). the logistic model (or logit
model) is a statistical model that models the probability of an event taking place by having
the log-odds for the event be a linear combination of one or more independent variables.

K-Nearest Neighbors : It is a machine learning algorithm based on Supervised Learning
technique. In k-NN classification, the output is a class membership. An object is classified
by a plurality vote of its neighbors, with the object being assigned to the class most
common among its k nearest neighbors (k is a positive integer, typically small). If k = 1,
then the object is simply assigned to the class of that single nearest neighbor.
In k-NN regression, the output is the property value for the object. This value is the average
of the values of k nearest neighbors.

Decision Tree : are a type of Supervised Machine Learning where the data is continuously
split according to a certain parameter. The tree can be explained by two entities, namely
decision nodes and leaves. The leaves are the decisions or the final outcomes. And the
decision nodes are where the data is split.

XGB Regressor : is a boosting algorithm based on gradient boosted decision trees
algorithm. It applies better regularization from technique to reduce overfitting, and it is one
of the differences from the gradient boosting.

Random Forest : This regression algorithm uses a technique that integrates predictions
from various machine learning algorithms to get a more accurate prediction than a single
machine learning model. During training, a Random Forest constructs many decision trees
and outputs the mean of all the classes involved as the final prediction of all the trees. There
is no interaction between the decision trees as they run in parallel to each other and perform
their learning.

Ridge Regression: a method of estimating the coefficients of multiple-regression models
in scenarios where the independent variables are highly correlated.


# Performance metrics:

R² Score (Coefficient of Determination):
R^2 = 1 - (SS_res / SS_tot)

where:
SS_res = ∑(y_i - ŷ_i)²
SS_tot = ∑(y_i - ȳ)²

MAE (Mean Absolute Error):
MAE = (1 / n) * ∑|y_i - ŷ_i|

where:
y_i = Actual value
ŷ_i = Predicted value
n = Number of observations

RMSE (Root Mean Squared Error):
RMSE = √((1 / n) * ∑(y_i - ŷ_i)²)

where:
y_i = Actual value
ŷ_i = Predicted value
n = Number of observations

Sum of Squares of Residuals (SS_res):
SS_res = ∑(y_i - ŷ_i)²

where:
y_i = Actual value
ŷ_i = Predicted value

Total Sum of Squares (SS_tot):
SS_tot = ∑(y_i - ȳ)²

where:
y_i = Actual value
ȳ = Mean of the actual values

