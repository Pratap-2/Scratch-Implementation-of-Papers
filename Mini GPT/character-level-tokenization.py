import numpy as np

rng = np.random.default_rng(42)

text="""
the cat sat on the mat.
the cat ate the fish.
the dog sat on the mat.
the dog ate the food.
language models learn patterns from text.
a model predicts the next character.
"""

chars = sorted(set(text))

#dictionary
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)

#encoding
data = np.array([stoi[ch] for ch in text],dtype = np.int64)

#decoding 
def decode(token_ids):
    return "".join(itos[int(i)] for i in token_ids)

block_size = 16
def get_batch(batch_size = 4):
    starts = rng.integers(
        0,
        len(data) - block_size,
        size= batch_size
    )

    x = np.stack([data[s:s+block_size] for s in starts])

    y=np.stack([data[s+1:s+block_size+1] for s in starts])

    return x,y


if __name__ == "__main__":
    X,y = get_batch()
    print("Input shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nInput:", decode(X[0]))
    print("Target:", decode(y[0]))



    print("Vocabulary size:", vocab_size)
    print("Number of characters:", len(data))
    print("Vocabulary:", "".join(chars))
    print("First 30 token IDs:", data[:30])
    print("Decoded text:", decode(data[:30]))

