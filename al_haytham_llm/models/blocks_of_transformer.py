import torch
import torch.nn as nn
from haytham_llm.models.attention import MultiHeadAttention
from haytham_llm.models.feedforward import FeedForward



class TransformerBlock(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.attention = MultiHeadAttention(config)
        self.feedforward = FeedForward(config)
        
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        
        self.dropout_layer = nn.Dropout(config.dropout)
        
        
    def forward (self ,x, kv_cache, use_cache ):
        
        attention_output , updated_cache = self.attention(
            self.layer_norm1(x) , kv_cache, use_cache
        )
        
        x += self.dropout_layer(attention_output)
        
        feedforward_output = self.feedforward(self.layer_norm2(x))
        
        x += self.dropout_layer(feedforward_output)
        
        
        
        return x , updated_cache