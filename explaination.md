# Transformer concepts and questions

This note answers the questions added to `rewire.md` and relates them to the current `encoder.py` and `decoder.py`. The examples use NumPy and two-dimensional, unbatched arrays, matching the shape convention in those files. They illustrate the math; they do not make NumPy parameters trainable.

## 1. Should each stacked layer be a separate object? Does seeding solve backpropagation?

**Yes: make one layer object per stack position, and make each object's parameters once.** A Transformer stack is a sequence of layers with the same *structure*, not the same weights. For example, with three encoder layers:

```text
input -> EncoderLayer(weights 1) -> EncoderLayer(weights 2)
      -> EncoderLayer(weights 3) -> output
```

In `encoder.py`, `stack_encoders` calls `self.encoder_layer(X)` repeatedly. There is only one `Encoder` object, so all iterations use the same `wq`, `wk`, `wv`, `W0`, FFN weights, and LayerNorm parameters. The current decoder similarly owns one set of its parameters. The architecture normally gives each layer its own parameters.

Create the layer list once during model construction, not inside `stack_encoders` or `stack_decoders`:

```python
rng = np.random.default_rng(seed)
self.layers = [
    EncoderLayer(d_model=d_model, num_heads=num_heads, rng=rng)
    for _ in range(stack_sz)
]
```

Then the forward pass only applies the already-created objects:

```python
for layer in self.layers:
    X = layer(X)
```

**Do not reseed or recreate the layers on each forward pass.** A seed makes initialization reproducible; it does not preserve learning. If weights are recreated on every pass, an optimizer's updates are discarded, so training cannot accumulate. Seed one generator once and draw different arrays from it for each layer. This gives repeatable runs with distinct parameter values.

There is an important limitation here: **the current code is NumPy-only, so it has no automatic differentiation/backpropagation or optimizer.** A list of NumPy layer objects fixes the architectural parameter-sharing issue but does not add training. For actual gradient training, use a framework such as PyTorch and create each layer once in `__init__` (for example, as entries in `torch.nn.ModuleList`). PyTorch then registers the parameters and tracks gradients. The seed controls the initial random values; the framework's optimizer updates those values.

## 2. How does dropout work, and where should it go?

Dropout randomly zeros elements during training to discourage over-reliance on a small set of activations. With inverted dropout, retained values are divided by the keep probability. That scaling makes the expected activation stay the same, so no rescaling is needed at inference time.

A small NumPy helper for a forward calculation is:

```python
def dropout(x, probability, rng, training):
    if not training or probability == 0:
        return x
    keep_probability = 1.0 - probability
    mask = rng.random(x.shape) < keep_probability
    return x * mask / keep_probability
```

Validate that `0 <= probability < 1` when configuring the model. Use the same model-owned generator, and expose a `training` flag so inference is deterministic:

```python
X = dropout(X, self.dropout_probability, self.rng, training)
```

In the original Transformer, dropout is used:

- On the sum of token embeddings and positional encodings.
- On attention probabilities after softmax (attention dropout).
- On each attention/FFN sublayer's output before adding the residual.

For a post-normalized encoder block, the latter two placements look conceptually like:

```python
attention_output = attention(...)
X = layer_norm(X + dropout(attention_output, p, rng, training))

ffn_output = feed_forward(X)
X = layer_norm(X + dropout(ffn_output, p, rng, training))
```

Attention dropout is applied to the softmax weights before they multiply `V`; it is separate from dropout on the final attention output. Dropout masks are sampled anew on each training forward pass, but the model's weight arrays remain the same. This helper is suitable for a small NumPy demonstration, not a training implementation: NumPy still will not calculate gradients.

## 3. Padding is not used; the decoder causal mask is separate

These NumPy blocks do not implement padding masks. Give the encoder and decoder unpadded sequences: every row in the input arrays should represent a real token embedding. If two sentences have different lengths, process them separately rather than adding `<PAD>` rows.

The decoder still uses a **causal/look-ahead mask** in target self-attention. This is not a padding mask: it prevents a target position from seeing later target positions, so the model cannot use future answers while predicting the current token. Cross-attention does not use this causal mask because encoder memory is source context.

If you later batch examples by padding shorter ones, you will need padding masks again so attention ignores the artificial positions.

## 4. What does reproducible, suitable initialization mean?

Initialization chooses starting values for trainable weights. If weights are too large or too small, activations and gradients can become unstable as they pass through many layers. A typical simple approach is Xavier/Glorot scaling for linear projections and He scaling for a layer followed by ReLU.

Using a generator consistently makes all initial values reproducible:

```python
rng = np.random.default_rng(seed)

# Example Xavier/Glorot normal initialization for a linear projection.
limit = np.sqrt(2.0 / (fan_in + fan_out))
W = rng.standard_normal((fan_in, fan_out)) * limit

# Example He normal initialization before ReLU.
W_relu = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
```

Use `rng.standard_normal` for the FFN too; the current `initialize_ffn` calls global `np.random.randn`, which is separate from the seeded generator. Avoid resetting the same seed before every layer: doing so can give identically shaped layers identical starting weights. Pass the already-seeded generator into each layer and draw from it in sequence.

The current FFN uses a He-style factor, which is reasonable for its ReLU, but its weights are not drawn from the object's seeded generator. The attention matrices and output projection are unscaled standard-normal values, so their scale is not controlled. These are initialization concerns, distinct from the fact that the NumPy implementation cannot train.

## 5. Should decoder cross-attention also be multi-headed?

**Yes.** In the original Transformer, all three decoder sublayers use multi-head structure where applicable: masked target self-attention, encoder-decoder (cross) attention, and FFN (the FFN is position-wise, not attention).

For each cross-attention head:

- Queries `Q` come from the current decoder hidden states.
- Keys `K` and values `V` come from the encoder output ("memory").
- Each of `Q`, `K`, and `V` has its own learned projection for that head.
- The attention score shape is `(target_length, source_length)`, which is allowed to be rectangular because source and target lengths need not match.

For a two-dimensional, unbatched implementation, one head can be written as:

```python
q = decoder_states @ Wq       # (target_length, d_k)
k = encoder_memory @ Wk       # (source_length, d_k)
v = encoder_memory @ Wv       # (source_length, d_k)

scores = q @ k.T / np.sqrt(d_k)  # (target_length, source_length)
weights = softmax(scores)        # row-wise, shape unchanged
head = weights @ v               # (target_length, d_k)
```

Repeat this with separate `Wq`, `Wk`, and `Wv` for every head; concatenate heads along the feature axis, then apply a learned output projection `W0`:

```python
joined = np.concatenate(heads, axis=-1)  # (target_length, d_model)
cross_output = joined @ W0               # (target_length, d_model)
```

Do not use the decoder self-attention's projections for cross-attention. Cross-attention is a different sublayer with its own learned projections. Pass `encoder_memory` explicitly to the decoder layer/stack, rather than relying on an attribute that was never initialized.

## 6. Current decoder errors to correct

These are based on the current file, not the earlier version of the notes:

1. **Odd-width check uses matrix multiplication.** In `positional_embedding`, `if (d_model @ 2 != 0)` uses Python's matrix-multiplication operator `@`, not modulo. A runtime check with `d_model=4` raised:
   ```text
   TypeError: unsupported operand type(s) for @: 'int' and 'int'
   ```
   To reject odd widths, use the remainder test `d_model % 2 != 0`. Alternatively, implement the last unpaired sine term so odd widths are supported.

2. **Decoder softmax still returns a 3D result.** With scores of shape `(2, 2)`, the current `softmax` returns shape `(2, 2, 1)`, because `rowmax[i]` is a one-element array. Then `sft_output @ V` cannot match the final dimension of the weights to the first dimension of `V`. Subtract the scalar `rowmax[i, 0]` in the loop, or use a vectorized 2D implementation:
   ```python
   exp_score = np.exp(score - np.max(score, axis=-1, keepdims=True))
   return exp_score / np.sum(exp_score, axis=-1, keepdims=True)
   ```

3. **Encoder memory is still missing.** `combine_decoder` reads `self.encoder_output`, but the current constructor does not create it, and `stack_decoders` does not accept it. Pass `encoder_memory` to `stack_decoders` and through each `combine_decoder` call to cross-attention. Otherwise, once positional encoding and self-attention work, cross-attention raises `AttributeError` for the missing `encoder_output`.

4. **Padding masks are intentionally not implemented.** These blocks expect inputs without padded rows. The decoder's causal mask remains enabled for target self-attention; it prevents looking at future target positions and is not a padding mask.

5. **Layer weights are still shared across stack iterations.** `stack_decoders` repeatedly calls `self.combine_decoder`, using the same object's parameters every time. A list of independently initialized decoder-layer objects is needed for independent weights per layer; creating/reseeding layers on every forward pass is not the solution.

## 7. How do the encoder and decoder connect?

The encoder transforms the source sequence once:

```text
source embeddings + positions -> encoder stack -> encoder_memory
```

The decoder consumes a target prefix plus that memory:

```text
shifted target embeddings + positions
    -> decoder stack(target states, encoder_memory)
    -> vocabulary projection -> next-token logits
```

The encoder memory is used as `K` and `V` in every decoder cross-attention layer; the current decoder hidden states supply `Q`. There can be different source and target sequence lengths. This is why the decoder needs both inputs. A token embedding and vocabulary projection can live in a separate model file as planned; the encoder/decoder blocks can keep accepting and returning `(sequence_length, d_model)` arrays.

## 8. A note about what "exact Transformer" entails

Fixing the layer stack and attention details makes the blocks closer to the paper, but the current files still use unbatched NumPy arrays and have no tokenizer, token embedding tables, vocabulary logits, loss, optimizer, or autoregressive generation loop. They are forward-pass learning exercises, not a complete trainable sequence-to-sequence Transformer. A PyTorch version is needed for ordinary automatic backpropagation and parameter updates.
