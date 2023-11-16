# Consider the following observations/data. And apply simple linear regression and findout estimated coefficients b0 and b1.(use numpypackage)
# x=[0,1,2,3,4,5,6,7,8,9,11,13]
# y = ([1, 3, 2, 5, 7, 8, 8, 9, 10, 12,16, 18]

import numpy as np
# Data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 16, 18])

# Calculate the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the differences between each data point and the mean
x_diff = x - mean_x
y_diff = y - mean_y

# Calculate b1 (slope) by taking the dot product of x_diff and y_diff divided by the dot product of x_diff with itself
b1 = np.sum(x_diff * y_diff) / np.sum(x_diff *x_diff)

# Calculate b0 (intercept) using the formula b0 = mean(y) - b1 * mean(x)
b0 = mean_y - b1 * mean_x

# Print the estimated coefficients
print("Estimated Coefficient b0 (Intercept):", b0)
print("Estimated Coefficient b1 (Slope):", b1)
