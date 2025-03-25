import torch
from .evaluator import HandEvaluator
from .model import PokerNN
import psutil
import time

class MacPokerBot:
    def __init__(self, model_path=None):
        self.evaluator = HandEvaluator()
        self.model = PokerNN()
        self.device = self._get_device()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
    
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def get_action(self, hole, community, pot, current_bet):
        start = time.time()
        
        # Thermal check
        if psutil.sensors_temperatures().get('coretemp', [0])[0].current > 85:
            time.sleep(0.5)
        
        strength = self.evaluator.monte_carlo_sim(hole, community)
        pot_odds = current_bet / (pot + current_bet) if pot > 0 else 0
        
        state = torch.tensor([
            strength,
            pot_odds,
            len(community)/5,
            current_bet/1000,
            torch.rand(1).item()
        ], device=self.device)
        
        with torch.no_grad():
            q_values = self.model(state)
            
        action = torch.argmax(q_values).item()
        print(f"Decision in {time.time()-start:.2f}s")
        return ['fold', 'call', 'raise'][action]