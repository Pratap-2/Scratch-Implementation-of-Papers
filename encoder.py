import numpy as np

class Encoder:

    def __init__(self,num_heads,stack_sz,d_model,dropout_probability=0.0):
        if num_heads <= 0 or stack_sz <= 0 or d_model <= 0:
            raise ValueError("num_heads, stack_sz, and d_model must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= dropout_probability < 1:
            raise ValueError("dropout_probability must be in [0, 1)")

        self.random_seed = 42
        self.num_heads  =  num_heads
        self.stack_sz = stack_sz
        self.d_model = d_model
        self.dropout_probability = dropout_probability
        self.rng = np.random.default_rng(self.random_seed)
        rng = self.rng
        self.d_k = d_model // num_heads
        self.eps = 1e-5
        self.d_ff = 4 * self.d_model

        self.layers = []
        for _ in range(stack_sz):
            # Each stack layer gets its own weights, initialized once.
            self.layers.append({
                "wk": [rng.standard_normal((d_model, self.d_k))
                       * np.sqrt(2 / (d_model + self.d_k))
                       for _ in range(num_heads)],
                "wq": [rng.standard_normal((d_model, self.d_k))
                       * np.sqrt(2 / (d_model + self.d_k))
                       for _ in range(num_heads)],
                "wv": [rng.standard_normal((d_model, self.d_k))
                       * np.sqrt(2 / (d_model + self.d_k))
                       for _ in range(num_heads)],
                "W0": rng.standard_normal((d_model, d_model))
                      * np.sqrt(2 / (2 * d_model)),
                "w1": rng.standard_normal((d_model, self.d_ff))
                      * np.sqrt(2 / self.d_model),
                "b1": np.zeros((1, self.d_ff)),
                "w2": rng.standard_normal((self.d_ff, d_model))
                      * np.sqrt(2 / (self.d_ff + d_model)),
                "b2": np.zeros((1, d_model)),
                "gamma1": np.ones(d_model),
                "beta1": np.zeros(d_model),
                "gamma2": np.ones(d_model),
                "beta2": np.zeros(d_model),
            })

        # Keep the original attributes available for the first layer.
        self.wk = self.layers[0]["wk"]
        self.wq = self.layers[0]["wq"]
        self.wv = self.layers[0]["wv"]
        self.W0 = self.layers[0]["W0"]
        self.w1 = self.layers[0]["w1"]
        self.b1 = self.layers[0]["b1"]
        self.w2 = self.layers[0]["w2"]
        self.b2 = self.layers[0]["b2"]
        self.gamma1 = self.layers[0]["gamma1"]
        self.beta1 = self.layers[0]["beta1"]
        self.gamma2 = self.layers[0]["gamma2"]
        self.beta2 = self.layers[0]["beta2"]

    def dropout(self, X, training=False):
        if not training or self.dropout_probability == 0:
            return X
        keep_probability = 1 - self.dropout_probability
        mask = self.rng.random(X.shape) < keep_probability
        return X * mask / keep_probability

    def softmax(self,score):
        "Aims to compute the softmax of the vector"
        maxscore = np.max(score,axis=-1,keepdims=True)
        exp_scores = np.exp(score - maxscore)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def scaled_dot_product(self,K,Q,V,training=False):
        "Aims to help compute scaled dot product of three vectors"

        score = Q @ K.T
        self.d_k = K.shape[1]
        score /= np.sqrt(self.d_k)

        score_softmax = self.softmax(score)
        score_softmax = self.dropout(score_softmax, training)
        ans = score_softmax @ V

        return score_softmax,ans

    def positional_encoding(self,num_seq,X):
        "Aims to add the position info in the embedding"
        if(self.d_model % 2 !=0 ) :
            raise ValueError("The embedding should be even in length")

        pos_embed = [[0.0] * self.d_model for _ in range(num_seq)]

        for pos in range(num_seq):
            for i in range(0,self.d_model,2):
                denom = 10000 ** (i / self.d_model)
                pos_embed[pos][i] = np.sin(pos/denom)
                pos_embed[pos][i+1] = np.cos(pos/denom)

        X = X + np.array(pos_embed)
        return X

    def self_attention(self,X,head_axis,layer_index=0,training=False):
        "Aims to compute the attention scores using K,Q,V blocks"

        layer = self.layers[layer_index]
        wq = layer["wq"][head_axis]
        wk = layer["wk"][head_axis]
        wv = layer["wv"][head_axis]

        k = X @ wk
        v = X @ wv
        q = X @ wq

        attention , output= self.scaled_dot_product(
            K=k,Q=q,V=v,training=training
        )

        return np.array(attention) , np.array(output)

    def multiple_head_attention(self,X,layer_index=0,training=False):
        "Aims to combine multiple self attention heads"
        layer = self.layers[layer_index]
        heads_value = []
        for i in range(self.num_heads):
            heads_value.append(self.self_attention(
                X,i,layer_index,training=training
            ))

        full_head = np.concatenate(
            [scaled_ans for attention,scaled_ans in heads_value],
            axis=-1
        ) #each scaled_ans has dim (seq_len,d_k) so merge along column axis

        return np.array(full_head @ layer["W0"]) # (seq_len,d_model)

    def layer_normalization(self,X,gamma,beta):
        "aims to normalize the rows"
        X = np.asarray(X,dtype=np.float32)

        #Normalize
        mean = np.mean(X,axis=-1,keepdims=True)
        var  = np.var(X,axis=-1,keepdims=True)

        X_norm = (X-mean) / np.sqrt(var + self.eps)

        return gamma * X_norm + beta 

    def add_and_norm(self,X,sublayer_output,gamma,beta):
        "Aims to do the element wise addition"
        residual =  X + sublayer_output
        return self.layer_normalization(residual,gamma,beta)

    def initialize_ffn(self,d_model,d_ff):
        self.w1 = self.rng.standard_normal((d_model,d_ff)) * np.sqrt(2/self.d_model)
        self.b1 = np.zeros((1,d_ff))

        self.w2 = self.rng.standard_normal((d_ff,d_model)) * np.sqrt(2/self.d_ff)
        self.b2 = np.zeros((1,d_model))
        return

    def feedforwardnn(self,X,layer_index=0):
        #first linear transformation 
        layer = self.layers[layer_index]
        hidden = X @ layer["w1"] + layer["b1"]

        # RELU activation
        hidden = np.maximum(0,hidden)

        #second layer transformation 
        output = hidden @ layer["w2"]  + layer["b2"]

        return output

    def encoder_layer(self, X,layer_index=0,training=False):
        layer = self.layers[layer_index]
        # 1. Multi-head self-attention
        attention_output = self.multiple_head_attention(
            X,layer_index,training=training
        )

        # 2. Residual connection + first LayerNorm
        Z = self.add_and_norm(
            X, self.dropout(attention_output,training),
            layer["gamma1"], layer["beta1"]
        )

        # 3. Feed-forward network
        ffn_output = self.feedforwardnn(Z,layer_index)

        # 4. Residual connection + second LayerNorm
        output = self.add_and_norm(
            Z, self.dropout(ffn_output,training),
            layer["gamma2"], layer["beta2"]
        )

        return output

    def stack_encoders(self,X,training=False):
        X = self.positional_encoding(X.shape[0],X=X)
        X = self.dropout(X,training)
        for layer_index in range(self.stack_sz):
            X = self.encoder_layer(X,layer_index,training=training)

        return X

if __name__ == "__main__":
    encoder = Encoder(
        num_heads=4,
        stack_sz=1,
        d_model=64
    )

    rng = np.random.default_rng(7)
    X = rng.standard_normal((10, 64))

    output = encoder.stack_encoders(X)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)
    print("All finite:", np.isfinite(output).all())
