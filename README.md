# Transformer Encoder–Decoder (NumPy)

This folder contains a learning implementation of the encoder and decoder blocks from the Transformer architecture in *Attention Is All You Need*. It demonstrates a forward pass on already-created embedding vectors. It is **not yet a complete text-to-text Transformer** and it does not train.

## What currently works

- The encoder accepts a 2D array of source embeddings with shape `(source_length, d_model)`.
- Sinusoidal positional encodings are added to the source embeddings.
- Encoder layers perform multi-head self-attention, residual addition and LayerNorm, then a position-wise ReLU feed-forward network, followed by another residual addition and LayerNorm.
- The decoder accepts a 2D array of target embeddings plus the encoder's output (encoder memory).
- Decoder layers perform causal masked target self-attention, multi-head cross-attention over encoder memory, then a position-wise feed-forward network. Each sublayer has its own residual connection and LayerNorm.
- Source and target lengths may differ. The example tests use six source positions and four target positions.
- Separate parameter sets are initialized for each layer in a stack. A fixed random seed makes initialization repeatable.
- Dropout can be enabled with `dropout_probability` and `training=True`. It defaults to `0.0` and is inactive unless explicitly requested.
- Padding masks are not implemented. Inputs must contain real embeddings only; do not append padding rows. The decoder's causal mask is implemented and remains enabled.

## Run the integration tests

From this directory:

```bash
../.venv/bin/python test_transformer.py
```

The tests use synthetic vectors, not tokenized sentences. They check that:

1. Encoder memory flows into the decoder and produces the expected shapes.
2. Repeated inference with dropout off gives the same output.
3. Changing a future target position does not change earlier decoder outputs.
4. Softmax returns a 2D attention matrix whose rows sum to one.

The tests passing means these focused forward-pass checks pass; it does not establish that the implementation is a complete or trainable Transformer.

## Minimal usage

`source_embeddings` and `target_embeddings` must already be numeric embedding arrays. Tokenization and token embedding lookup are not part of these classes.

```python
import numpy as np
from encoder import Encoder
from decoder import decoder

d_model = 8
encoder = Encoder(num_heads=2, stack_sz=2, d_model=d_model)
decoder_model = decoder(d_model=d_model, num_heads=2, stack_sz=2)

rng = np.random.default_rng(7)
source_embeddings = rng.standard_normal((6, d_model))
target_embeddings = rng.standard_normal((4, d_model))

encoder_memory = encoder.stack_encoders(source_embeddings)
decoder_states = decoder_model.stack_decoders(
    target_embeddings,
    encoder_memory,
)

print(encoder_memory.shape)  # (6, 8)
print(decoder_states.shape)  # (4, 8)
```

## Progress and contribution estimate

These percentages are approximate project-planning estimates, not objective measurements:

- **Encoder/decoder block forward pass:** roughly 70–80% of the small NumPy block-level goal is implemented.
- **Complete, trainable text-to-text Transformer:** roughly 25–30% is implemented. The embedding/data path, vocabulary output, loss, training, and generation are still missing.
- **Your contribution to the current work:** estimated at about **50%**, based on the conversation: you wrote the initial encoder/decoder, made edits yourself, and chose the learning/design direction; assistant collaboration added or corrected pieces such as attention-shape handling, stack parameter separation, multi-head cross-attention, masks/dropout options, and tests. This is a rough shared-work estimate, not a line-by-line authorship count. There is no version history here to calculate an exact percentage.

## Files

- `encoder.py` — encoder block and stack.
- `decoder.py` — decoder block and stack, including cross-attention.
- `test_transformer.py` — focused integration and behavior tests.
- `explaination.md` — learning explanations and current limitations.


## Made by 
Aditya Pratap Singh