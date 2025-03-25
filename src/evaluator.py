from numba import njit
from treys import Evaluator, Deck
import numpy as np

class HandEvaluator:
    def __init__(self):
        self.evaluator = Evaluator()
    
    @njit
    def monte_carlo_sim(self, hole, community, iterations=500):
        deck = Deck()
        [deck.draw(1) for _ in range(2)]  # Remove hole cards
        wins = 0
        
        for _ in range(iterations):
            opp_hole = deck.draw(2)
            remaining = 5 - len(community)
            sim_community = community + deck.draw(remaining)
            
            our_score = self.evaluator.evaluate(sim_community, hole)
            opp_score = self.evaluator.evaluate(sim_community, opp_hole)
            
            if our_score < opp_score:
                wins += 1
            deck.shuffle()
            
        return wins / iterations