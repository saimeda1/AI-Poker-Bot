import torch
import torch.nn as nn

class PokerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    def forward(self, x):
        return self.net(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = PokerNN()
    model.load_state_dict(torch.load(path))
    return model