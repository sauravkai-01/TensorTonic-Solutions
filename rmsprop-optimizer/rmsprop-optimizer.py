import numpy as np


def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    g_arr = np.asarray(g, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)

    new_s_arr = beta * s_arr + (1.0 - beta) * (g_arr * g_arr)
    new_w_arr = w_arr - (lr / (np.sqrt(new_s_arr) + eps)) * g_arr

    new_w = np.round(new_w_arr, 6).tolist()
    new_s = np.round(new_s_arr, 6).tolist()

    return new_w, new_s