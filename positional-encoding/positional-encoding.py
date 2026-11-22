import numpy as np


def positional_encoding(
    seq_len: int, d_model: int, base: float = 10000.0
) -> np.ndarray:
  """
    Returns a NumPy array of shape (seq_len, d_model) with sinusoidal positional encodings.
    """
  # Positions: shape (seq_len, 1)
  positions = np.arange(seq_len, dtype=np.float64)[:, np.newaxis]

  # Number of frequency pairs needed (ceil(d_model / 2))
  num_freqs = (d_model + 1) // 2

  # Dimension indices i for each frequency pair: shape (1, num_freqs)
  i = np.arange(num_freqs, dtype=np.float64)[np.newaxis, :]

  # Denominator base^(2i / d_model): shape (1, num_freqs)
  div_term = np.power(base, (2.0 * i) / d_model)

  # Scaled angles: shape (seq_len, num_freqs)
  angles = positions / div_term

  # Allocate output matrix with float dtype: shape (seq_len, d_model)
  pe = np.zeros((seq_len, d_model), dtype=np.float64)

  # Even indices (0, 2, 4, ...) get sin; odd indices (1, 3, 5, ...) get cos
  pe[:, 0::2] = np.sin(angles)

  # When d_model is odd, there is one fewer cos column than sin column
  num_cos = d_model // 2
  if num_cos > 0:
    pe[:, 1::2] = np.cos(angles[:, :num_cos])

  return pe