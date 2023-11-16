#Consider the student data set.It can be downloaded from(https://drive.google.com/open?id=1oakZCv7g3mlmCSdv9J8kdSaqO5_6dIOw). Write a programme in python to apply simple linear regression and find out mean absolute error, mean squared error and root mean squared error

# Linear Regression is a type of supervised learning algorithm used for predicting a continuous outcome variable (also called the dependent variable) based on one or more predictor variables (independent variables).

# Import necessary libraries
import numpy as np  # NumPy is a library for numerical operations in Python
from sklearn.linear_model import LinearRegression  # Linear Regression model from scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Metrics for model evaluation

# Create input features (X) and target variable (y) as NumPy arrays
X = np.array([2.5, 5.1, 3.2, 8.5, 3.5]).reshape(-1, 1)  # Reshape to a column vector
y = np.array([21, 47, 27, 75, 30])

# Create a Linear Regression model
model = LinearRegression()

# Train (fit) the Linear Regression model on the provided data
model.fit(X, y)   # This is where the model "learns" from the data

# Use the trained model to make predictions on the input features (X)
y_pred = model.predict(X)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)

#The code implements a simple linear regression model using the scikit-learn library in Python.
#It generates a dataset with input features (`X`) and target variable (`y`), trains the linear regression model, makes predictions, and calculates evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess the model's performance on the provided data.
