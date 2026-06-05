from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        _, (hidden, _) = self.lstm2(x)
        x = self.dropout2(hidden[-1])
        return self.head(x)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(self.dropout(hidden[-1]))

