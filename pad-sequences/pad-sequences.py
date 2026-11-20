import numpy as np


def pad_sequences(
    seqs: list, pad_value: int = 0, max_len: int | None = None
) -> np.ndarray:
  """
    Returns: np.ndarray of shape (N, L) where:
        N = len(seqs)
        L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
  num_seqs = len(seqs)
  if num_seqs == 0:
    return np.empty((0, 0), dtype=int)

  target_len = (
      max_len if max_len is not None else max(len(seq) for seq in seqs)
  )
  out = np.full((num_seqs, target_len), pad_value, dtype=int)

  for i, seq in enumerate(seqs):
    length = min(len(seq), target_len)
    if length > 0:
      out[i, :length] = seq[:length]

  return out