import numpy as np

class decoder:

    def __init__(self,d_model,num_heads,stack_sz,dropout_probability=0.0):
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.stack_sz = stack_sz
        self.random_seed = 42

        self.rng = np.random.default_rng(self.random_seed)
        rng = self.rng
        self.d_k = d_model // self.num_heads
        self.eps = 1e-5
        self.d_ff = 4 * self.d_model

        self.layers = []
        for _ in range(stack_sz):
            # Self-attention and cross-attention need separate projections.
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
                "cross_wk": [rng.standard_normal((d_model, self.d_k))
                             * np.sqrt(2 / (d_model + self.d_k))
                             for _ in range(num_heads)],
                "cross_wq": [rng.standard_normal((d_model, self.d_k))
                             * np.sqrt(2 / (d_model + self.d_k))
                             for _ in range(num_heads)],
                "cross_wv": [rng.standard_normal((d_model, self.d_k))
                             * np.sqrt(2 / (d_model + self.d_k))
                             for _ in range(num_heads)],
                "cross_W0": rng.standard_normal((d_model, d_model))
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
                "gamma3": np.ones(d_model),
                "beta3": np.zeros(d_model),
            })

        # Keep the original attributes available for the first layer.
        for name in self.layers[0]:
            setattr(self, name, self.layers[0][name])
        self.dropout_probability = dropout_probability

    def dropout(self, X, training=False):
        if not training or self.dropout_probability == 0:
            return X
        keep_probability = 1 - self.dropout_probability
        mask = self.rng.random(X.shape) < keep_probability
        return X * mask / keep_probability

    def softmax(self,score):
        score = np.asarray(score,dtype=np.float32)
        rowmax = np.max(score,axis=-1,keepdims=True)
        
        exp_score = np.exp(score - rowmax)
        return exp_score / np.sum(exp_score,axis=-1,keepdims=True)

    def positional_embedding(self,X):
        num_seq , d_model = X.shape
        if d_model % 2 != 0:
            raise ValueError("Not possible with odd dimensions")
        
        pos_embed = [[0.0] * d_model  for _ in range(num_seq)]

        for pos in range(num_seq):
            for i in range(0,d_model,2):
                denom = 10000 ** (i/d_model)
                pos_embed[pos][i] = np.sin(pos/denom)
                pos_embed[pos][i+1] = np.cos(pos/denom)

        f_embed = X + pos_embed
        return np.array(f_embed)

    def scaled_dot_product(self,Q,K,V,mask=False,training=False):
        output = Q @ K.T
        output/= np.sqrt(K.shape[1]) 
        if mask:
            # Prevent target positions from attending to later target tokens.
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    if j > i:
                        output[i, j] = -1e9
        sft_output = self.softmax(output)
        sft_output = self.dropout(sft_output,training)
        f_output = sft_output @ V

        return np.array(f_output)

    def selfattention(self,X,idx,mask=False,layer_index=0,training=False):

        layer = self.layers[layer_index]
        k = X @ layer["wk"][idx]
        q = X @ layer["wq"][idx]
        v = X @ layer["wv"][idx]

        return self.scaled_dot_product(
            q,k,v,mask,training=training
        )

    def multiheadattention(self,X,mask=False,layer_index=0,training=False):
        layer = self.layers[layer_index]
        all_heads = []

        for i in range(self.num_heads):
            all_heads.append(self.selfattention(
                X,i,mask=mask,layer_index=layer_index,
                training=training
            ))

        output = np.concatenate(all_heads,axis=-1)

        return np.array(output @ layer["W0"])

    def cross_multiheadattention(self,X,encoder_memory,layer_index=0,training=False):
        layer = self.layers[layer_index]
        all_heads = []
        for i in range(self.num_heads):
            q = X @ layer["cross_wq"][i]
            k = encoder_memory @ layer["cross_wk"][i]
            v = encoder_memory @ layer["cross_wv"][i]
            all_heads.append(self.scaled_dot_product(
                q,k,v,training=training
            ))
        joined = np.concatenate(all_heads,axis=-1)
        return joined @ layer["cross_W0"]

    def add_x_and_output(self,x,attention_output,gamma,beta):
        residual =  x+attention_output

        return self.layer_normalize(residual,gamma,beta)

    def layer_normalize(self,x,gamma,beta):
        "aims to normalize the rows"
        x = np.asarray(x,dtype=np.float32)

        mean = np.mean(x,axis=-1,keepdims=True)
        var  = np.var(x,axis=-1,keepdims=True)

        x_norm = (x-mean) / np.sqrt(var + self.eps)
        x_final = gamma *x_norm   + beta 

        return np.array(x_final)

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


    def combine_decoder(self,x,encoder_memory,layer_index=0,training=False):
        layer = self.layers[layer_index]
        #step-1
        X = self.multiheadattention(
            x,mask=True,layer_index=layer_index,training=training
        )
        X_res1 = self.add_x_and_output(
            x,self.dropout(X,training),layer["gamma1"],layer["beta1"]
        )

        #step-2
        x_cross_attention = self.cross_multiheadattention(
            X_res1,encoder_memory,layer_index,training=training
        )
        X_res2 = self.add_x_and_output(
            x=X_res1,attention_output=self.dropout(x_cross_attention,training),
            gamma=layer["gamma2"],beta=layer["beta2"]
        )

        #step3
        X_ffn = self.feedforwardnn(X_res2,layer_index)
        x_res3= self.add_x_and_output(
            x=X_res2,attention_output=self.dropout(X_ffn,training),
            gamma=layer["gamma3"],beta=layer["beta3"]
        )

        return x_res3

    def stack_decoders(self,X,encoder_memory,training=False):
        X = np.asarray(X)
        encoder_memory = np.asarray(encoder_memory)
       
        X_pos = self.positional_embedding(X)
        X_pos = self.dropout(X_pos,training)
        for layer_index in range(self.stack_sz):
            X_pos = self.combine_decoder(
                X_pos,encoder_memory,layer_index,training=training
            )

        return X_pos
