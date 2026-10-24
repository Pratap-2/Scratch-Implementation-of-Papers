import numpy as np


class TokenPositionEmbedding:
    def __init__(self, vocab_size, max_seq_len, embedding_dim, rng):

        # One trainable vector per vocabulary token.
        # Shape: (vocab_size, embedding_dim)
        self.token_embedding_weights = (
            rng.standard_normal((vocab_size, embedding_dim)) * 0.02
        )

        # One trainable vector per possible sequence position.
        # Shape: (max_seq_len, embedding_dim)
        self.position_embedding_weights = (
            rng.standard_normal((max_seq_len, embedding_dim)) * 0.02
        )

        # Gradients of the loss with respect to the two weight matrices.
        # These start at zero and are filled during backward().
        self.token_embedding_gradients = np.zeros_like(
            self.token_embedding_weights
        )
        self.position_embedding_gradients = np.zeros_like(
            self.position_embedding_weights
        )

        # Saved from forward() because backward() needs to know
        # which token IDs were looked up in the forward pass.
        self.input_token_ids = None
        self.input_sequence_length = None

    def forward(self, input_token_ids):
        """
        Convert token IDs into vectors and add positional information.

        input_token_ids: (batch_size, sequence_length)
        Returns:         (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length = input_token_ids.shape

        # Every input position must have a corresponding
        # positional embedding.
        if sequence_length > self.position_embedding_weights.shape[0]:
            raise ValueError("Sequence exceeds maximum supported length")

        # Positions are 0, 1, 2, ..., sequence_length - 1.
        position_ids = np.arange(sequence_length)

        # Look up the vector for each input token.
        # Shape: (B, T, D)
        token_vectors = self.token_embedding_weights[input_token_ids]

        # Look up one positional vector for each sequence position.
        # Shape: (T, D)
        position_vectors = self.position_embedding_weights[position_ids]

        # Add a batch dimension so NumPy can broadcast the same
        # positional vectors across every sequence in the batch.
        # (B, T, D) + (1, T, D) -> (B, T, D)
        hidden_states = token_vectors + position_vectors[None, :, :]

        # Cache the inputs needed to calculate gradients later.
        self.input_token_ids = input_token_ids.copy()
        self.input_sequence_length = sequence_length

        return hidden_states

    def backward(self, gradient_from_next_layer):
        """
        Accumulate gradients for the token and positional embeddings.

        gradient_from_next_layer is dL/d(hidden_states).
        Shape: (B, T, D)
        """

        # Start fresh for this backward pass.
        # The optimizer will update the weights using these gradients.
        self.token_embedding_gradients.fill(0.0)
        self.position_embedding_gradients.fill(0.0)

        # Each token occurrence contributes its gradient to the
        # corresponding row of the token embedding matrix.
        # Repeated token IDs must have their gradients summed.
        # Accumulate gradients for token embeddings

        for batch_idx in range(self.input_token_ids.shape[0]):
            for pos_idx in range(self.input_token_ids.shape[1]):
                token_id = self.input_token_ids[batch_idx, pos_idx]

                self.token_embedding_gradients[token_id] += (
                    gradient_from_next_layer[batch_idx, pos_idx]
                )

        # Accumulate gradients for positional embeddings across the batch
        for pos_idx in range(self.input_sequence_length):
            for batch_idx in range(self.input_token_ids.shape[0]):
                self.position_embedding_gradients[pos_idx] += (
                    gradient_from_next_layer[batch_idx, pos_idx]
                )


        # Token IDs are discrete integers, so no gradient is returned for them
        return None