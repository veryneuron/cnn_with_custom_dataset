from torch import nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.6
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(7 * 7 * 64, 128, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        self.fc2 = nn.Linear(128, 26, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc2(out)

        return out