import torch
from torch import nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, 1, bias=False, dtype=torch.float64)

    def forward(self, x):
        x = self.linear(x)
        return x.repeat(1, self.out_dim)


class NeuralNetwork_with_Mu(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, 1, bias=False, dtype=torch.float64)
        self.mu = nn.Linear(1, out_dim, bias=False, dtype=torch.float64)

    def forward(self, x):
        x = self.linear(x)
        x = self.mu(x)
        return x


class NeuralNetworkTwoHiddenReluEnd(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer1 = nn.Linear(
            in_dim, hidden_dim, bias=True, dtype=torch.float64
        )
        self.hidden_layer2 = nn.Linear(
            hidden_dim, hidden_dim, bias=True, dtype=torch.float64
        )
        self.output_layer = nn.Linear(
            hidden_dim, out_dim, bias=True, dtype=torch.float64
        )
        self.gelu = (
            nn.GELU()
        )  # nn.sequential(nn.Linear(hidden_dim_1, hidden_dim_1, bias=True, dtype=torch.float64), nn.GELU()) #nn.ReLU() (we could just have one hidden layer)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.gelu(x)
        x = self.hidden_layer2(x)
        x = self.gelu(x)
        x = self.output_layer(x)
        return x


class CircularMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 16,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_layer1 = nn.Linear(
            in_dim, hidden_dim, bias=True, dtype=torch.float64
        )
        self.output_layer = nn.Linear(
            hidden_dim, out_dim, bias=True, dtype=torch.float64
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


class MNISTSimpleNet(nn.Module):
    def __init__(self):
        super(MNISTSimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # 'fc' stands for fully connected layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output layer: 10 neurons for digits 0-9

    def forward(self, x):
        # This defines how data flows through the network
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleCIFARCNN(nn.Module):
    def __init__(self):
        super(SimpleCIFARCNN, self).__init__()
        # 3 input channels (RGB), 16 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # After two pools, the 32x32 image becomes 8x8.
        # 32 channels * 8 * 8 = 2048 input features for the linear layer
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
