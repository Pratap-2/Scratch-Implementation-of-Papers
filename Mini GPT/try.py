import numpy as np

def ssoftmax(score):
    max_row = [max(row) for row in score]
    s1 = []
    for i,row in enumerate(score):
        nrow = np.array([np.exp(val - max_row[i]) for val in row])
        s1.append(nrow)
    exp_sums = [sum(row) for row in s1]
    s2=[]
    for i,row in enumerate(s1):
        nrow = np.array([val/exp_sums[i] for val in row])
        s2.append(nrow)
    return s2


score = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 2.0, 2.0],
    [4.0, 1.0, 0.0]
])

p = scaled_softmax(score)

print(p)
print(np.sum(p,axis=-1))

def self_attention(X,rng):
    d_model = X.shape[1]
    wk = rng.standard_normal((d_model,d_model),dtype = np.float32)
    wq = rng.standard_normal((d_model,d_model),dtype = np.float32)
    wv = rng.standard_normal((d_model,d_model),dtype = np.float32)

    k = X @ wk
    q = X @ wq
    v = X @ wv

    attention = q @ k.T 
    attention /= np.sqrt(k.shape[1])

    softmax = scaled_softmax(attention)

    foutput = softmax @ v

    return foutput


def causal_implementation(previous_layer_output,rng,dim_k):
    d_model = previous_layer_output.shape[1]
    wk = rng.standard_normal((d_model,dim_k),dtype = np.float32)
    wq = rng.standard_normal((d_model,dim_k),dtype = np.float32)
    wv = rng.standard_normal((d_model,dim_k),dtype = np.float32)

    k = previous_layer_output @ wk
    q = previous_layer_output @ wq
    v = previous_layer_output @ wv

    attention = q @ k.T 
    attention /= np.sqrt(k.shape[1])
    
    T = previous_layer_output.shape[0]
    causal_mask = np.tril(
        np.ones((T, T),dtype=bool),
    )

    attention = np.where(causal_mask,attention,-1e9)

    softmax = scaled_softmax(attention)
    
    foutput = softmax @ v

    return foutput


def causal_multihead_attention(previous_layer_output,num_heads,rng):
    num_seq , d_model = previous_layer_output.shape

    if(d_model % num_heads != 0):
        raise ValueError("Not possible")

    dim_k = d_model // num_heads 
    # for now dimk = dimv

    outputs = []
    for _ in range(num_heads):
        outputs.append(causal_implementation(previous_layer_output=previous_layer_output,rng=rng,dim_k=dim_k))
    final_output = np.concatenate(outputs,axis=-1)

    w = rng.standard_normal((d_model,d_model),dtype = np.float32)

    return final_output @ w


rng = np.random.default_rng(42)

X = rng.standard_normal((5, 8), dtype=np.float32)

output = causal_multihead_attention(
    previous_layer_output=X,
    num_heads=2,
    rng=rng
)

print(output.shape)