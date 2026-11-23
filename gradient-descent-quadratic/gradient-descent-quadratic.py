def gradient_descent_quadratic(
    a: float, b: float, c: float, x0: float, lr: float, steps: int
) -> float:
    """Returns the final scalar x after the requested iterations."""
    x = float(x0)

    for _ in range(steps):
        # Derivative of f(x) = a*x^2 + b*x + c with respect to x
        grad = 2.0 * a * x + b
        x = x - lr * grad

    return float(x)