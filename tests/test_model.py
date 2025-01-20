import pytest
from chess_board_recognizer.model import CNNModel
import torch


def test_model():
    model = CNNModel()
    x = torch.randn(16, 3, 128, 128)
    y = model(x)
    assert y.shape == (16, 8, 8, 13)

@pytest.mark.parametrize("tensor", [torch.randn((16, 3, 128)), torch.randn((16, 3, 3, 128, 128))])
def test_error_on_wrong_shape(tensor):
    model = CNNModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(tensor)
