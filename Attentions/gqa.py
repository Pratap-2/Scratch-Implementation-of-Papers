import numpy as np
from mha import causal_attention
from mqa import MultiQueryAttention
class GroupedQueryAttention:
    def __init__(self,d_model,num_query_heads,num_kv_heads,seed=42):
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")

        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        
        self.num_kv_heads = num_kv_heads
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads

        rng = np.random.default_rng(seed)
        self.wq = rng.standard_normal(
            (d_model,num_query_heads*self.head_dim),dtype=np.float32
        )
        self.wk = rng.standard_normal(
            (d_model, num_kv_heads * self.head_dim), dtype=np.float32
        )
        self.wv = rng.standard_normal(
            (d_model, num_kv_heads * self.head_dim), dtype=np.float32
        )
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

        k = k.reshape(
            T, self.num_kv_heads, self.head_dim
        )
        v = v.reshape(
            T, self.num_kv_heads, self.head_dim
        )

        head_outputs = []
        for h in range(self.num_query_heads):

            kv_head = h // self.group_size

            q_h = q[: , h , :]
            k_h = k[: , kv_head, :]
            v_h = v[: , kv_head, :]

            out_h = causal_attention(q_h,k_h,v_h)
            head_outputs.append(out_h)

        combined = np.concatenate(head_outputs,axis=-1)

        output = combined @ self.wo

        return output



rng = np.random.default_rng(7)

T = 5
d_model = 8
num_query_heads = 4

x = rng.normal(size=(T, d_model))

mqa = MultiQueryAttention(
    d_model=d_model,
    num_query_heads=num_query_heads
)

gqa = GroupedQueryAttention(
    d_model=d_model,
    num_query_heads=num_query_heads,
    num_kv_heads=2
)

mqa_output = mqa.forward(x)
gqa_output = gqa.forward(x)

print("Input shape:", x.shape)
print("MQA output shape:", mqa_output.shape)
print("GQA output shape:", gqa_output.shape)

assert mqa_output.shape == (T, d_model)
assert gqa_output.shape == (T, d_model)