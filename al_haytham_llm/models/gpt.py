
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from al_haytham_llm.models.blocks_of_transformer import TransformerBlock
from al_haytham_llm.models.embeddings import InputEmbedding


class GPTModel(nn.Module):
    """
    GPT language model.

    Input:
        input_ids: (batch_size, sequence_length)

    Output:
        logits: (batch_size, sequence_length, vocab_size)

    If targets are provided, also returns the next-token loss.
    If use_cache=True, returns updated KV caches for generation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Converts token IDs into token + positional embeddings.
        # Input:  (B, T)
        # Output: (B, T, d_model)
        self.embedding = InputEmbedding(config)

        # Stack of Transformer blocks.
        # Each block preserves the shape: (B, T, d_model).
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )

        # Normalizes the final hidden representations.
        # Input / Output: (B, T, d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

        # Projects each token representation to vocabulary scores.
        # Input:  (B, T, d_model)
        # Output: (B, T, vocab_size)
        self.language_model_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # Share token embedding weights with the output projection.
        # Both layers use the same weight matrix.
        if config.tie_weights:
            self.language_model_head.weight = (
                self.embedding.token_embedding.weight
            )

        # Initialize model parameters, then scale residual projections.
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module):
        """Initialize Linear and Embedding weights."""

        if isinstance(module, nn.Linear):
            # Weight shape: (out_features, in_features)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # Initialize biases to zero when present.
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Embedding weight shape: (num_embeddings, embedding_dim)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        """Scale residual-output projections following GPT-2-style init."""

        # Scale factor depends on the number of Transformer blocks.
        scale = math.sqrt(2 * self.config.n_layers)

        for block in self.blocks:
            # Modify weights without tracking gradients.
            with torch.no_grad():
                # Attention output: (B, T, d_model)
                block.attention.output_projection.weight.div_(scale)

                # FeedForward output: (B, T, d_model)
                block.feedforward.projection_layer.weight.div_(scale)

    def forward(
        self,
        input_ids,
        targets=None,
        kv_cache=None,
        use_cache=False,
    ):
        # input_ids: (B, T)
        # B = batch size, T = number of input tokens.

        # Each layer's cache contains:
        # (key_states, value_states)
        # Each tensor: (B, n_heads, T_past, d_k)
        #
        # Read the number of cached tokens to continue positional indexing.
        cached_sequence_length = (
            kv_cache[0][0].size(2) if kv_cache is not None else 0
        )

        # Convert token IDs into embeddings.
        # Input:  (B, T)
        # Output: (B, T, d_model)
        hidden_states = self.embedding(
            input_ids,
            position_offset=cached_sequence_length,
        )

        # Store updated caches only when caching is enabled.
        updated_kv_caches = [] if use_cache else None

        # Process hidden states through every Transformer block.
        # Input / Output per block: (B, T, d_model)
        for layer_index, block in enumerate(self.blocks):

            # Retrieve this block's previous cache, if available.
            current_kv_cache = (
                kv_cache[layer_index] if kv_cache is not None else None
            )

            # Apply attention and FeedForward operations.
            hidden_states, updated_kv_cache = block(
                hidden_states,
                use_cache=use_cache,
                kv_cache=current_kv_cache,
            )

            # Save the updated cache for this layer.
            if use_cache:
                updated_kv_caches.append(updated_kv_cache)

        # Normalize the final token representations.
        # Input / Output: (B, T, d_model)
        hidden_states = self.final_layer_norm(hidden_states)

        # Produce a score for every vocabulary token at every position.
        # Output: (B, T, vocab_size)
        logits = self.language_model_head(hidden_states)

        # During inference without targets, return logits and cache.
        if targets is None:
            return logits, None, updated_kv_caches

        # Flatten batch and sequence dimensions for cross-entropy.
        # logits:  (B*T, vocab_size)
        # targets: (B*T)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.config.pad_id,
        )

        # Return predictions, training loss, and updated caches.
        return logits, loss, updated_kv_caches

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Return the total number of model parameters."""

        total = sum(parameter.numel() for parameter in self.parameters())

        # Optionally exclude token and positional embedding tables.
        if non_embedding:
            total -= self.embedding.position_embedding.weight.numel()
            total -= self.embedding.token_embedding.weight.numel()

        return total
