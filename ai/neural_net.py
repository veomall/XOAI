import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.fc1 = nn.Linear(board_size * board_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, board_size * board_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, self.board_size * self.board_size)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

