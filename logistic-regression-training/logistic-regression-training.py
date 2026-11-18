import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Returns elementwise sigmoid values."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """Returns the trained weights and bias as (w, b)."""
    N, D = X.shape
    
    # 1. Initialize weights w as zeros and bias b as 0.0
    w = np.zeros(D)
    b = 0.0
    
    # 2. Gradient descent optimization loop
    for _ in range(steps):
        # Linear combination
        z = np.dot(X, w) + b
        
        # Predictions (sigmoid)
        p = _sigmoid(z)
        
        # Gradients for binary cross-entropy loss
        # X.T @ (p - y) has shape (D,), scaled by 1/N
        grad_w = np.dot(X.T, (p - y)) / N
        grad_b = np.mean(p - y)
        
        # Update parameters
        w -= lr * grad_w
        b -= lr * grad_b
        
    return w, b