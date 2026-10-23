import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        self.expansion_layer = nn.Linear(config.d_model,config.d_ff)
        self.activation_function = nn.GELU(approximate="tanh") # GELU بدل ما اقل من 0 ب زيرو بسيب مارجين بسيط 
        self.projection_layer = nn.Linear(config.d_ff,config.d_model) # fc is fully connected

        
        
    def forward(self, hidden_states):
        
        hidden_states = self.expansion_layer(hidden_states)
        hidden_states = self.activation_function(hidden_states)
        return self.projection_layer(hidden_states)
        
        
