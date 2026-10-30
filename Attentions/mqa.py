import numpy as np
from mha import causal_attention

class MultiQueryAttention:
    def __init__(self,d_model,num_query_heads,seed=42):
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.head_dim = d_model // num_query_heads

        rng = np.random.default_rng(seed)
        self.wq = rng.standard_normal((d_model,num_query_heads*self.head_dim),dtype=np.float32)
        self.wk = rng.standard_normal((d_model,self.head_dim),dtype=np.float32)
        self.wv = rng.standard_normal((d_model,self.head_dim),dtype=np.float32)
        # Combine all Query-head outputs
        self.wo = rng.standard_normal((d_model,d_model),dtype=np.float32)


    def forward(self,x):
        """
        x: (T, d_model)
        returns: (T, d_model)
        """

        T = x.shape[0]

        # 1. Project input
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        # 2. Split Query into multiple heads
        q = q.reshape(
            T, self.num_query_heads, self.head_dim
        )
        # q: (T, Hq, head_dim)
        # k and v remain single shared heads
        # k: (T, head_dim)
        # v: (T, head_dim)

        # 3. Every Query head uses the SAME K and V
        head_outputs = []

        for h in range(self.num_query_heads):
            q_h = q[:,h,:]
            out_h = causal_attention(q_h,k,v)
            head_outputs.append(out_h)

        combined_q = np.concatenate(head_outputs,axis=-1)

        output = combined_q @ self.wo

        return output


    