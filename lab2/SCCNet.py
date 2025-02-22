import torch
import torch.nn as nn

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        self.spatialConv = nn.Conv2d(1, Nu, kernel_size=(C, 1), bias=False)
        self.batchNorm1 = nn.BatchNorm2d(Nu)
        self.squareLayer = SquareLayer()
        self.avgPool = nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 6))
        self.logLayer = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropoutRate)

        self.temporalConv = nn.Conv2d(Nu, Nc, kernel_size=(1, Nt))
        self.batchNorm2 = nn.BatchNorm2d(Nc)

        fc_size = self.get_size(Nc, timeSample)
        self.fc = nn.Linear(fc_size, numClasses)

    def forward(self, x):
        x = self.spatialConv(x)
        x = self.batchNorm1(x)
        x = self.squareLayer(x)
        x = self.avgPool(x)
        x = torch.log(x)
        x = self.dropout(x)

        x = self.temporalConv(x)
        x = self.batchNorm2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_size(self, Nc, timeSample):
        size = ((timeSample - 1) // 6 + 1 - 1 + 1)
        return Nc * size