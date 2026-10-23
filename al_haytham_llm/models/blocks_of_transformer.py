import torch
import torch.nn as nn
from al_haytham_llm.models.attention import MultiHeadAttention
from al_haytham_llm.models.feedforward import FeedForward



class TransformerBlock(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.attention = MultiHeadAttention(config)
        self.feedforward = FeedForward(config)
        
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        
        self.dropout_layer = nn.Dropout(config.dropout)
        
        
    def forward(self, hidden_states, kv_cache, use_cache):
        
        attention_output, updated_kv_cache = self.attention(
            self.layer_norm1(hidden_states), kv_cache, use_cache
        )
        
        hidden_states += self.dropout_layer(attention_output)
        
        feedforward_output = self.feedforward(self.layer_norm2(hidden_states))
        
        hidden_states += self.dropout_layer(feedforward_output)
        
        return hidden_states, updated_kv_cache
