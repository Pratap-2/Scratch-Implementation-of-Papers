import numpy as np


def causal_attention(q, k, v):
    """
    q: (T, head_dim)
    k: (T, head_dim)
    v: (T, head_dim)

    returns: (T, head_dim)
    """

    T, head_dim = q.shape

    # 1. Calculate attention scores
    scores = q @ k.T / np.sqrt(head_dim)
    # Shape: (T, T)

    # 2. Apply causal mask
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores[mask] = -np.inf

    # 3. Softmax, row-wise
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(
        attention_weights, axis=-1, keepdims=True
    )

    # 4. Weighted sum of Values
    output = attention_weights @ v
    # Shape: (T, head_dim)

    return output