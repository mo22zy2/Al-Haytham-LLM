import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.seq_len=config.seq_len
        
        self.dropout= nn.Dropout(config.dropout) #Droput Layer
        
        self.token_embedding= nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        self.position_embedding = nn.Embedding(
            config.seq_len,
            config.d_model
        )
        
        
        
    def forward(self, input_ids, position_offset:int=0):
        
        _ , sequence_length = input_ids.shape
        
        if sequence_length > self.seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds"
                f"seq_len = {self.seq_len}"
            )
            
        positions = torch.arange(
            position_offset,
            position_offset+sequence_length,
            device=input_ids.device
        )
        
        tokens=self.token_embedding(input_ids)
        
        position_vectors = self.position_embedding(positions)
        
        return self.dropout(tokens+position_vectors) # (B,seq_len,d_model)
