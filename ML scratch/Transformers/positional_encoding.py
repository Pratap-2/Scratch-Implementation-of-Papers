import numpy as np


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model // 2):
            denominator = 10000 ** (2 * i / d_model)

            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE


# Example: 5 tokens, each with a 6-dimensional embedding
PE = positional_encoding(seq_len=5, d_model=6)

np.set_printoptions(precision=4, suppress=True)
print(PE)
print("Shape:", PE.shape)
