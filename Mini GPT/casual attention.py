import numpy as np
from embeddings import TokenPositionEmbedding
from character_level_tokenization import *

class CausalSelfAttention:
    def __init__(self, embedding_dim, rng):
        self.embedding_dim = embedding_dim

        # Trainable projection matrices.
        # Each transforms an input vector of size D into a vector of size D.
        self.query_weights = (
            rng.standard_normal((embedding_dim, embedding_dim)) * 0.02
        )
        self.key_weights = (
            rng.standard_normal((embedding_dim, embedding_dim)) * 0.02
        )
        self.value_weights = (
            rng.standard_normal((embedding_dim, embedding_dim)) * 0.02
        )

        # Gradients of the loss with respect to these matrices.
        self.query_gradients = np.zeros_like(self.query_weights)
        self.key_gradients = np.zeros_like(self.key_weights)
        self.value_gradients = np.zeros_like(self.value_weights)

        # Saved values needed during backpropagation.
        self.input_hidden_states = None
        self.query_vectors = None
        self.key_vectors = None
        self.value_vectors = None
        self.attention_weights = None
        self.causal_mask = None

    def forward(self, hidden_states):
        """
        hidden_states: (B, T, D)
        returns:       (B, T, D)
        """
        batch_size, sequence_length, embedding_dim = hidden_states.shape

        # Save input because gradients of the projection weights
        # depend on the original input.
        self.input_hidden_states = hidden_states

        # Project input vectors into Query, Key and Value vectors.
        self.query_vectors = hidden_states @ self.query_weights  # (B,T,D)
        self.key_vectors = hidden_states @ self.key_weights      # (B,T,D)
        self.value_vectors = hidden_states @ self.value_weights  # (B,T,D)

        # Compute similarity between every query and every key.
        # Scores[b, i, j] measures how much position i attends to j.
        attention_scores = (
            self.query_vectors @ self.key_vectors.transpose(0, 2, 1)
        ) / np.sqrt(embedding_dim)  # (B,T,T)

        # Allow a position to attend only to itself and earlier positions.
        self.causal_mask = np.tril(
            np.ones((sequence_length, sequence_length), dtype=bool)
        )
        print(self.causal_mask)
        # Set forbidden future-position scores to a very negative number.
        attention_scores = np.where(
            self.causal_mask[None, :, :], # jaha pe ye true vaha pe attention_score ko vahi rahne do varna -1e9 kardo 
            attention_scores,
            -1e9
        )

        # Stable softmax: normalize scores into probabilities.
        shifted_scores = (
            attention_scores
            - np.max(attention_scores, axis=-1, keepdims=True)
        )

        exponentials = np.exp(shifted_scores)
        self.attention_weights = exponentials / np.sum(
            exponentials, axis=-1, keepdims=True
        )

        # Weighted combination of Value vectors.
        output_hidden_states = (
            self.attention_weights @ self.value_vectors
        )  # (B,T,D)

        return output_hidden_states

    def backward(self, gradient_from_next_layer): #(g=\partial L/\partial O) g -> gradient  from next layer
        """
        gradient_from_next_layer: dL/d(output_hidden_states), shape (B,T,D)

        Calculates gradients for Q/K/V projection matrices
        and returns dL/d(input_hidden_states).
        """
        hidden_states = self.input_hidden_states
        query = self.query_vectors
        key = self.key_vectors
        value = self.value_vectors
        weights = self.attention_weights

        embedding_dim = hidden_states.shape[-1]
        scale = np.sqrt(embedding_dim)

        # Step 1: output = attention_weights @ value
        gradient_attention_weights = (
            gradient_from_next_layer @ value.transpose(0, 2, 1)
        )
        gradient_value = (
            weights.transpose(0, 2, 1) @ gradient_from_next_layer
        )

        # Step 2: softmax backward.
        # For softmax output p and incoming gradient g:
        # d_scores = p * (g - sum(g * p))
        gradient_scores = weights * (
            gradient_attention_weights
            - np.sum(
                gradient_attention_weights * weights,
                axis=-1,
                keepdims=True
            )
        )

        # The masked future positions must receive zero score gradient.
        gradient_scores = np.where(
            self.causal_mask[None, :, :],
            gradient_scores,
            0.0
        )

        # Step 3: scores = query @ key.T / sqrt(D)
        gradient_query = (
            gradient_scores @ key
        ) / scale

        gradient_key = (
            gradient_scores.transpose(0, 2, 1) @ query
        ) / scale

        # Step 4: gradients of the three projection matrices.
        # Sum contributions across batch and sequence positions.
        flattened_inputs = hidden_states.reshape(-1, embedding_dim)

        self.query_gradients = (
            flattened_inputs.T @ gradient_query.reshape(-1, embedding_dim)
        )
        self.key_gradients = (
            flattened_inputs.T @ gradient_key.reshape(-1, embedding_dim)
        )
        self.value_gradients = (
            flattened_inputs.T @ gradient_value.reshape(-1, embedding_dim)
        )

        # Step 5: pass the gradient back to the input.
        gradient_hidden_states = (
            gradient_query @ self.query_weights.T
            + gradient_key @ self.key_weights.T
            + gradient_value @ self.value_weights.T
        )

        return gradient_hidden_states  

if __name__ == "__main__":
    x, y = get_batch(batch_size=4)

    embedding = TokenPositionEmbedding(
        vocab_size=vocab_size,
        max_seq_len=block_size,
        embedding_dim=16,
        rng=rng
    )

    attention = CausalSelfAttention(
        embedding_dim=16,
        rng=rng
    )

    # Forward pass
    hidden_states = embedding.forward(x)
    attention_output = attention.forward(hidden_states)

    print("Input IDs:", x.shape)
    print("Embedding output:", hidden_states.shape)
    print("Attention output:", attention_output.shape)

    # Test backward mechanics with a dummy gradient.
    dummy_gradient = rng.standard_normal(attention_output.shape)
    gradient_to_embeddings = attention.backward(dummy_gradient)

    print("Gradient to embeddings:", gradient_to_embeddings.shape)
    print("Query weight gradients:", attention.query_gradients.shape)