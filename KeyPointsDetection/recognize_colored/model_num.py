import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 10, 5, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(10, 20, 5, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.BatchNorm2d(20),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8000, 6),

        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x