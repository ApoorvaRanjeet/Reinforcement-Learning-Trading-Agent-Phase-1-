import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # second hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == '__main__':
    state_dim  = 4    # price_vs_ma20, price_change, volume_vs_avg, holding
    action_dim = 3    # buy, sell, hold
    net    = DQN(state_dim, action_dim)
    state  = torch.randn(1, state_dim)
    output = net(state)
    print(f"Input shape:  {state.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Q-values: {output}")
    # should print 3 Q-values — one per action