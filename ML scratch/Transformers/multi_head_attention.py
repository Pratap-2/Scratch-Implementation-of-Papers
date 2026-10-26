import numpy as np
import math

def softmax(score):
    rowmax= np.max(score,axis=-1,keepdims=True)
    score = score-rowmax
    score=np.exp(score)
    return score/np.sum(score,axis=-1,keepdims = True)


def softmaxnumpy(score):

    rowmax=[max(row) for row in score]
    exp_scores = []
    for i,row in enumerate(score):
        mxe = rowmax[i]
        exp_row = [math.exp(val-mxe) for val in row]
        exp_scores.append(exp_row)
    

    sumrow = [sum(row) for row in exp_scores]

    attention_weights =[]
    for i,row in enumerate(exp_scores):
        sum_row = sumrow[i]
        norm_row =[val/sum_row for val in row]
        attention_weights.append(norm_row)
    return attention_weights


def scaled_dot_product(Q,K,V):
    dk = Q.shape[1]
    qkt = Q @ K.T
    qkt = qkt/math.sqrt(dk)

    sftm = softmaxnumpy(qkt)

    final_output = sftm @ V

    return final_output,sftm

def self_attention(X,dimk,dimv):
    d_model = X.shape[1]
    wk = np.random.randn(d_model,dimk)
    wq = np.random.randn(d_model,dimk)
    wv = np.random.randn(d_model,dimv)

    k = X @ wk
    q = X @ wq
    v = X @ wv

    final_output,sftm = scaled_dot_product(q,k,v)

    return final_output 

def multiheadattention(X,n_heads):

    n_samples,d_model = X.shape

    assert d_model % n_heads == 0

    heads = d_model // n_heads 

    allheads = []
    for i in range(n_heads):
        allheads.append(self_attention(X,dimk=heads,dimv=heads))

    final_output = np.concatenate(allheads,axis = -1)

    W = np.random.randn(d_model,d_model)

    return final_output @ W

X=np.random.randn(8,32)

n_heads = 4

final_ans = multiheadattention(X,n_heads=n_heads)

print(final_ans)
