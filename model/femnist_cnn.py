import torch
import torch.nn as nn


class FEMNIST_CNN(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN, self).__init__()

        self.covn1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # if stride=1, padding=(kernal_size-1)/2=(5-1)/2=2, #(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, ),  # (16,14,14)
        )
        self.covn2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 62)

    def forward(self, x):
        x = self.covn1(x)
        x = self.covn2(x)  # (batch,32,7,7)
        x = x.view(x.size(0), -1)  # (batch, 32*7*7)
        output = self.out(x)
        return output