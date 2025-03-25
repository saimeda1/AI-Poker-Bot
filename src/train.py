import torch
import torch.optim as optim
from .model import PokerNN, save_model

def train_model(epochs=100):
    model = PokerNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy training data (replace with real data)
    for epoch in range(epochs):
        # Example training loop
        inputs = torch.randn(32, 5)
        labels = torch.randint(0, 3, (32,))
        
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    
    save_model(model, "../data/trained_model.pth")