import numpy as np
import math

def softmax(scores):
    rowmax = [ max(row) for row in scores]

    exp_scores = []
    for i,row in enumerate(scores):
        row_max = rowmax[i]
        exp_row = [math.exp(val - row_max) for val in row]
        exp_scores.append(exp_row)

    rowsums = [sum(row) for row in exp_scores]

    attention_weights = []
    for i,row in enumerate(exp_scores):
        row_sum = rowsums[i]
        norm_row = [val/row_sum for val in row]
        attention_weights.append(norm_row)
    return attention_weights


def softmaxnumpy(scores):
    rowmax = np.max(scores,axis=-1,keepdims=True) # row ops
    scores = scores - rowmax

    exp_scores = np.exp(scores) # whole matrix pe exp laga do
    attention_weights = exp_scores/np.sum(exp_scores,axis =-1,keepdims=True)  #row ops
    return attention_weights


def scaled_dotproduct(Q,K,V):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)

    Returns:
        output: (seq_len, d_v)
        attention_weights: (seq_len, seq_len)
    """
    d_k = Q.shape[1]
    qkt = Q @ K.T
    qkt = qkt / math.sqrt(d_k)

    attention_weights = softmax(qkt)

    final_weights = attention_weights @ V

    return final_weights , attention_weights


def self_attention(X,dk,dv):
    """
    X: (seq_len, d_model)

    Returns:
        output: (seq_len, d_v)
        attention_weights: (seq_len, seq_len)
    """

    d_model = X.shape[1]
    wk = np.random.randn(d_model,dk) *0.1 #randn gives decimal numbers from normal numbers 
    wq = np.random.randn(d_model,dk) *0.1
    wv = np.random.randn(d_model,dv) *0.1


    k = X@wk
    q = X@wq
    v = X@wv

    # apply attention 
    final_weights,attention_weights = scaled_dotproduct(q,k,v)
    return final_weights,attention_weights








# 3 tokens, each represented by a 4-dimensional embedding
X = np.random.randn(3, 4)

output, weights = self_attention(X, dk=4, dv=4)

print("Input shape:", X.shape)
print("Output shape:", output.shape)

print("\nAttention weights:")
print(weights)

print("\nOutput:")
print(output)

print("\nRow sums:", np.sum(weights, axis=1))