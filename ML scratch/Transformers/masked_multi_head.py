import numpy as np
import math


def softmaxnumpy(score):

    rowmax = [max(row) for row in score]
    exp_scores = []

    for i, row in enumerate(score):
        mxe = rowmax[i]
        exp_row = [math.exp(val - mxe) for val in row]
        exp_scores.append(exp_row)

    sumrow = [sum(row) for row in exp_scores]

    attention_weights = []

    for i, row in enumerate(exp_scores):
        sum_row = sumrow[i]
        norm_row = [val / sum_row for val in row]
        attention_weights.append(norm_row)

    return np.array(attention_weights)

def tril(sequence):
    score = []
    n = sequence.shape[0]
    for i in range(n):
        row =[]
        for j in range(n):
            if(j<=i):
                row.append(1)
            else:
                row.append(0)
        score.append(row)
    return score


def where(qkt,mask):
    n =qkt.shape[0]
    nqkt=[]
    for i  in range(n):
        row=[]
        for j in range(n):
            if(mask[i][j]==1):
                row.append(qkt[i][j])
            else:
                row.append(-1e9)
        nqkt.append(row)
    return nqkt

def scaled_dot_product(Q, K, V, masked=True):

    dk = Q.shape[1]

    qkt = Q @ K.T
    qkt = qkt / math.sqrt(dk)

    # Causal masking
    if masked:
        seq_len = Q.shape[0]

        mask = tril(qkt)
        print("mask value:->",mask)
        # Future positions receive a very negative score
        qkt = where(qkt,mask)
        print("softmax after masking:",qkt)

    sftm = softmaxnumpy(qkt)

    final_output = sftm @ V

    print(sftm)
    return final_output, sftm


def self_attention(X, dimk, dimv, masked=True):

    d_model = X.shape[1]

    wk = np.random.randn(d_model, dimk)
    wq = np.random.randn(d_model, dimk)
    wv = np.random.randn(d_model, dimv)

    k = X @ wk
    q = X @ wq
    v = X @ wv

    final_output, sftm = scaled_dot_product(
        q, k, v, masked=masked
    )

    return final_output 


def multiheadattention(X, n_heads, masked=True):

    n_samples, d_model = X.shape

    assert d_model % n_heads == 0

    heads = d_model // n_heads

    allheads = []

    for i in range(n_heads):
        allheads.append(
            self_attention(
                X,
                dimk=heads,
                dimv=heads,
                masked=masked
            )
        )

    final_output = np.concatenate(allheads, axis=-1)

    W = np.random.randn(d_model, d_model)

    return final_output @ W


# Test
np.random.seed(42)
X = np.random.randn(8, 32)

n_heads = 4

final_ans = multiheadattention(
    X,
    n_heads=n_heads,
    masked=True
)

# print(final_ans.shape)
# print(final_ans)
