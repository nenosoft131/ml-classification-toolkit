import torch
from torch import nn as nn
from torch.nn import functional as F


class MLP1D(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1000):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.1, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.1, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin3(x), p=0.1, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin4(x), p=0.1, training=self.training)
        x = F.relu(x)
        x = torch.sigmoid(self.lin5(x))
        return x.squeeze(1).squeeze(1)

