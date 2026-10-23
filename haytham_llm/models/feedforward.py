import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        self.fc1 = nn.Linear(config.d_model,config.d_ff)
        self.act = nn.GELU(approximate="tanh") # GELU بدل ما اقل من 0 ب زيرو بسيب مارجين بسيط 
        self.fc2 = nn.Linear(config.d_ff,config.d_model) # fc is fully connected

        
        
    def forward(self,x):
        
        x= self.fc1(x)
        x= self.act(x)
        return self.fc2(x)
        
        