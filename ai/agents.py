import torch
import numpy as np

class AIAgent:
    def __init__(self, model, epsilon=0):
        self.model = model
        self.epsilon = epsilon

    def get_action(self, board):
        if np.random.random() < self.epsilon:
            return np.random.choice(board.size * board.size)
        else:
            state = torch.FloatTensor(board.get_state())
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()