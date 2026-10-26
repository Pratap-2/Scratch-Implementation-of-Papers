"""Input token IDs
      |
Token embeddings
      |
Transformer Layer 1
  ├── Multi-head attention
  │    ├── Q projection
  │    ├── K projection → Layer 1 K cache
  │    └── V projection → Layer 1 V cache
  └── Feed-forward network
      |
Transformer Layer 2
  ├── Multi-head attention
  │    ├── Q projection
  │    ├── K projection → Layer 2 K cache
  │    └── V projection → Layer 2 V cache
  └── Feed-forward network
      |
     ...
      |
Final normalization + vocabulary projection
      |
Next-token logits → sampling/selection"""

import numpy as np

class KVCache:
    def __init__(self,head_dim):
        self.head_dim = head_dim

        #Initially no KV cached tokens
        self.keys   = np.empty((0,head_dim))
        self.values = np.empty((0,head_dim))

    def append(self,new_key,new_value):
        # new value and new key have shape(head_dim,)
        new_key   = new_key.reshape(1,self.head_dim)
        new_value = new_value.reshape(1,self.head_dim)

        self.keys = np.concatenate(
            [self.keys,new_key] , axis = 0
        )
        self.values = np.concatenate(
            [self.values , new_value],axis = 0
        )
    
    def attention_step(self,x, Wq, Wk, Wv, cache):
        """
        x:   current token's hidden representation (d_model,)
        Wq:  (d_model, head_dim)
        Wk:  (d_model, head_dim)
        Wv:  (d_model, head_dim)

        Returns the attention output for the current token.
        """

        #project only the current token  
        q = x @ Wq 
        k = x @ Wk
        v = x @ Wv

        #2. append this token k and v
        cache.append(k,v)

        #3. attend over all the cache keys
        scores = cache.keys @ q
        scores = scores / np.sqrt(len(q))


        # Numerically stable softmax.
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / np.sum(weights)

        # 4. Weighted sum of all cached values.
        output = weights @ cache.values

        return output, weights


rng = np.random.default_rng(42)

d_model = 4
head_dim = 2 

wq = rng.standard_normal(size=(d_model,head_dim),dtype=np.float32)
wk = rng.standard_normal(size=(d_model,head_dim),dtype=np.float32)
wv = rng.standard_normal(size=(d_model,head_dim),dtype=np.float32)

cache = KVCache(head_dim=head_dim)

# Three token representations, for illustration.
tokens = rng.normal(size=(3, d_model))

for i, x in enumerate(tokens):
    output, weights = cache.attention_step(x, wq, wk, wv, cache)

    print(f"Token {i + 1}")
    print("Attention output:", output)
    print("Attention weights:", weights)
    print("K cache shape:", cache.keys.shape)
    print("V cache shape:", cache.values.shape)
    print("-" * 40)

