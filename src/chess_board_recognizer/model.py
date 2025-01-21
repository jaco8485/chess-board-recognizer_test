import torch
import torch.nn as nn
import timm
from loguru import logger


class CNNModel(nn.Module):
    """
    Simple CNN model. Takes in images of 128x128.
    """

    def __init__(self) -> nn.Module:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool3 = nn.MaxPool2d(2)

        # The size of the resulting image
        self.fc1 = nn.Linear(32 * 14 * 14, 2048)

        # A chess board has 8x8 squares and 13 possible states of a square.
        # 1 empty 6 white pieces and 6 black pieces.
        self.fc2 = nn.Linear(2048, 8 * 8 * 13)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            logger.error("CNNModel input was not a 4D Tensor")
            raise ValueError("Expected input to a 4D tensor")

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.pool3(x)
        x = self.activation(x)

        x = x.view(-1, 32 * 14 * 14)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)

        x = x.view(-1, 8, 8, 13)

        return x


class ResNet(nn.Module):
    """
    Simple CNN model. Takes in images of 128x128.
    """

    def __init__(self) -> nn.Module:
        super().__init__()

        self.model = timm.create_model("resnet18", pretrained=True, num_classes=8 * 8 * 13)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            logger.error("ResNet input was not a 4D Tensor")
            raise ValueError("Expected input to a 4D tensor")

        x = self.model(x)

        x = x.view(-1, 8, 8, 13)

        return x


if __name__ == "__main__":
    model = CNNModel()

    x = torch.randn((16, 3, 128, 128))

    output = model(x)
