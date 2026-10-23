import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from al_haytham_llm.models.blocks_of_transformer import TransformerBlock
from al_haytham_llm.models.embeddings import InputEmbedding


class GPTModel(nn.Module):
    """Embeddings, a stack of Transformer blocks, and a projection back to the
    vocabulary.

    Input:  (B, T) token IDs
    Output: (B, T, vocab_size) logits -- position t scores every possible
            token as the continuation of positions 0..t

    Pass `targets` to also get the next-token loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = InputEmbedding(config)

        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )

        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        self.language_model_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        if config.tie_weights:
            self.language_model_head.weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()


    def _init_weights(self, module):
        """GPT-2's initialisation: normal(0, 0.02), zeroed biases.

        PyTorch's Linear default depends on fan-in, which leaves a 768-wide
        model's activations larger than GPT-2 assumes and makes the early
        steps of training less stable.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        """Shrink the two projections that write into the residual stream.

        Every block adds twice into the residual stream, so with n_layers
        blocks the stream accumulates 2*n_layers contributions. Left alone its
        variance grows with depth. GPT-2 divides these weights by
        sqrt(2 * n_layers) so the stream stays at a stable scale however deep
        the model is.
        """
        scale = math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attention.output_projection.weight.div_(scale)
                block.feedforward.projection_layer.weight.div_(scale)

    def forward(self, input_ids, targets=None, kv_cache=None, use_cache=False):

        # kv_cache          # a list: one entry per layer -> [cache_0, cache_1, ... cache_11]
        # kv_cache[0]       # layer 0's cache = a tuple (k, v)
        # kv_cache[0][0]    # the k tensor
        # .size(2)          # k has shape (B, n_heads, T_past, d_k)
        #                   #               0    1       2     3    <- dim 2 is time

        cached_sequence_length = (
            kv_cache[0][0].size(2) if kv_cache is not None else 0
        )

        hidden_states = self.embedding(
            input_ids, position_offset=cached_sequence_length
        )  # (B, T, d_model)

        updated_kv_caches = [] if use_cache else None
        
        for layer_index, block in enumerate(self.blocks):
            current_kv_cache = (
                kv_cache[layer_index] if kv_cache is not None else None
            )
            hidden_states, updated_kv_cache = block(
                hidden_states,
                use_cache=use_cache,
                kv_cache=current_kv_cache,
            )
            if use_cache:
                updated_kv_caches.append(updated_kv_cache)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.language_model_head(hidden_states)

        if targets is None:
            return logits, None, updated_kv_caches

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.config.pad_id,
        )

        return logits, loss, updated_kv_caches

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Total trainable parameters.

        `non_embedding=True` excludes the token and position tables, which is
        the number usually quoted when comparing model sizes -- embeddings
        scale
        """

        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embedding.position_embedding.weight.numel()
            total -= self.embedding.token_embedding.weight.numel()

        return total