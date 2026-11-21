import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    if not A or not A[0]:
        return np.empty((0, 0))

    n_rows = len(A)
    n_cols = len(A[0])

    # Create a new array with swapped dimensions (M, N)
    result = np.empty((n_cols, n_rows), dtype=np.array(A).dtype)

    for i in range(n_rows):
        for j in range(n_cols):
            result[j, i] = A[i][j]

    return result