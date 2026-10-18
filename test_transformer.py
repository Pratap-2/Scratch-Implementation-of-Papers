import unittest

import numpy as np

from decoder import decoder
from encoder import Encoder


class TransformerConnectionTests(unittest.TestCase):

    def setUp(self):
        self.d_model = 8
        self.encoder = Encoder(num_heads=2, stack_sz=2, d_model=self.d_model)
        self.decoder = decoder(d_model=self.d_model, num_heads=2, stack_sz=2)

        rng = np.random.default_rng(7)
        # These are example embeddings; token embedding is outside these blocks.
        self.source_embeddings = rng.standard_normal((6, self.d_model))
        self.target_embeddings = rng.standard_normal((4, self.d_model))

    def test_encoder_memory_flows_through_decoder(self):
        memory = self.encoder.stack_encoders(self.source_embeddings)
        output = self.decoder.stack_decoders(self.target_embeddings, memory)

        self.assertEqual(memory.shape, (6, self.d_model))
        self.assertEqual(output.shape, (4, self.d_model))
        self.assertTrue(np.isfinite(memory).all())
        self.assertTrue(np.isfinite(output).all())

    def test_inference_is_repeatable(self):
        memory = self.encoder.stack_encoders(self.source_embeddings)
        first = self.decoder.stack_decoders(self.target_embeddings, memory)
        second = self.decoder.stack_decoders(self.target_embeddings, memory)

        np.testing.assert_allclose(first, second)

    def test_causal_mask_blocks_future_target_positions(self):
        memory = self.encoder.stack_encoders(self.source_embeddings)
        changed_target = self.target_embeddings.copy()
        changed_target[3] += 100

        original = self.decoder.stack_decoders(self.target_embeddings, memory)
        changed = self.decoder.stack_decoders(changed_target, memory)

        # Changing the last target position cannot affect earlier positions.
        np.testing.assert_allclose(original[:3], changed[:3], atol=1e-6)

    def test_attention_softmax_rows_sum_to_one(self):
        probabilities = self.decoder.softmax(
            np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
        )

        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
