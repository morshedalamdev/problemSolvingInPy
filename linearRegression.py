# Simple linear regression with NumPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Multiple linear regression with sklearn
from sklearn.linear_model import LinearRegression
from sklearn. model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Comparison of different regression techniques
from sklearn.linear_model import Ridge, Lasso


# ----------------------------------------
# Simple linear regression with NumPy
# ----------------------------------------
# Load the dataset
path = 'data/regress_data1.csv'
data = pd. read_csv(path)
# Read the CSV file containing population and revenue data

# Display first few rows
data.head()
# Shows the first 5 rows to understand data structure

# Cost function definition
def computeCost(X, y, w):
    """
    Compute the cost function J(w) = 1/(2m) * Σ(h(x) - y)²
    
    Parameters:
    X: feature matrix (m × n)
    y: target values (m × 1)
    w: weights/parameters (n × 1)
    
    Returns:
    cost: the computed cost
    """
    inner = np.power(((X * w. T) - y), 2)
    # Calculate (h(x) - y)² for all training examples
    # X * w.T gives predictions, subtract y, then square
    
    return np.sum(inner) / (2 * X.shape[0])
    # Sum all squared errors and divide by 2m
    # X. shape[0] gives number of training examples (m)

# Add bias term (column of ones)
data. insert(0, 'Ones', 1)
# Inserts a column of 1s at position 0 for the intercept term (w₀)

# Separate features and target
cols = data.shape[1]  # Get number of columns
X = data.iloc[:, : cols-1]  # All rows, all columns except last
y = data. iloc[:, cols-1:cols]  # All rows, only last column

# Convert to matrices
X = np.matrix(X. values)  # Convert DataFrame to numpy matrix
y = np.matrix(y.values)  # Convert DataFrame to numpy matrix
w = np.matrix(np.array([0, 0]))  # Initialize weights to zero

# Gradient descent implementation
def batch_gradientDescent(X, y, w, alpha, iters):
    """
    Perform batch gradient descent to learn θ
    
    Parameters: 
    X: feature matrix
    y: target values
    w: initial weights
    alpha: learning rate
    iters: number of iterations
    
    Returns:
    w: optimized weights
    cost: cost history
    """
    temp = np.matrix(np.zeros(w.shape))
    # Temporary matrix to store updated weights
    
    parameters = int(w.ravel().shape[1])
    # Get number of parameters (features + bias)
    
    cost = np.zeros(iters)
    # Array to store cost at each iteration
    
    for i in range(iters):
        # For each iteration
        
        error = (X * w.T) - y
        # Calculate prediction error:  h(x) - y
        
        for j in range(parameters):
            # For each parameter
            
            term = np.multiply(error, X[:, j])
            # Multiply error by j-th feature
            
            temp[0, j] = w[0, j] - ((alpha / len(X)) * np.sum(term))
            # Update rule: w := w - α * (1/m) * Σ(h(x) - y) * x
            
        w = temp
        # Update all weights simultaneously
        
        cost[i] = computeCost(X, y, w)
        # Store cost for this iteration
        
    return w, cost
    # Return optimized weights and cost history

# Set hyperparameters
alpha = 0.01  # Learning rate
iters = 1500  # Number of iterations

# Train the model
g, cost = batch_gradientDescent(X, y, w, alpha, iters)

# Calculate final cost
final_cost = computeCost(X, y, g)
print(f"Final cost: {final_cost}")

# Make predictions for visualization
x = np.linspace(data['population'].min(), data['population'].max(), 100)
# Create 100 evenly spaced points between min and max population

f = g[0, 0] + (g[0, 1] * x)
# Calculate predictions:  y = w₀ + w₁*x

# Plot results
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Predicted value')  # Predicted value
ax.scatter(data['population'], data['revenue'], label='Training data')  # Training data
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Revenue')
ax.set_title('Predicted Revenue vs Population')
plt.show()


# ----------------------------------------
# Multiple linear regression with sklearn
# ----------------------------------------

# Load Boston housing dataset from original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data_boston = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target_boston = raw_df.values[1::2, 2]

X = data_boston
y = target_boston

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")


# ----------------------------------------
# Comparison of different regression techniques
# ----------------------------------------

# Standard Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression R²: {lr.score(X_test, y_test):.4f}")

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge Regression R²: {ridge.score(X_test, y_test):.4f}")

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Lasso Regression R²: {lasso.score(X_test, y_test):.4f}")