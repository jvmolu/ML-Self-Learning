import numpy as np

class LinearRegressor:
    
    def __init__(self, lr=0.1):
        self.weights = None
    
    def fit(self, X, y, n_iter=1000, lr=0.1):
        # X - (m, n)
        # y - (m,)
        m, n = X.shape
        y = y.reshape((m, 1)) # FIX SHAPES
        # [Allows for multiple fit calls]
        if self.weights is None: # Initialize weights if not already initialized in previous fit
            # Random initialization
            self.weights = np.random.randn(n+1, 1)
        X = np.c_[np.ones((m, 1)), X]
        for _ in range(n_iter):
            # /m when using mean squared error, dont use /m when using sum squared error
            gradients = (2/m) * X.T.dot(X.dot(self.weights) - y)
            self.weights -= lr * gradients
        return self

    def predict(self, X):
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]
        return X.dot(self.weights)
    
    def mse(self, y_true, y_pred):
        y_true = y_true.reshape(len(y_true), 1)
        y_pred = y_pred.reshape(len(y_pred), 1)
        return np.mean((y_true - y_pred)**2)