
import numpy as np
import math

def softmax(score):
    score = score - np.max(score, axis=-1, keepdims=True)
    exp_score = np.exp(score)
    return exp_score / np.sum(exp_score, axis=-1, keepdims=True)


def scaled_dot_product(Q, K, V):
    dk = Q.shape[1]
    scores = (Q @ K.T) / math.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Initialize weights only ONCE
        self.Wq = []
        self.Wk = []
        self.Wv = []

        for i in range(n_heads):
            self.Wq.append(np.random.randn(d_model, self.head_dim))
            self.Wk.append(np.random.randn(d_model, self.head_dim))
            self.Wv.append(np.random.randn(d_model, self.head_dim))

        self.Wo = np.random.randn(d_model, d_model)

    def forward(self, X):
        all_heads = []

        for i in range(self.n_heads):
            Q = X @ self.Wq[i]
            K = X @ self.Wk[i]
            V = X @ self.Wv[i]

            output, weights = scaled_dot_product(Q, K, V)
            all_heads.append(output)

        concatenated = np.concatenate(all_heads, axis=-1)
        return concatenated @ self.Wo


X = np.random.randn(8, 32)

# Create the object once
attention = MultiHeadAttention(d_model=32, n_heads=4)

# Call forward multiple times
output1 = attention.forward(X)
output2 = attention.forward(X)

print(output1)
print(output2)


print(output1.shape)  # (8, 32)
print(np.allclose(output1, output2))  # True
