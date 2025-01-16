import pytest
from chess_board_recognizer.model import CNNModel
import torch


def test_model():
    model = CNNModel()
    x = torch.randn(16, 3, 128, 128)
    y = model(x)
    assert y.shape == (16, 8, 8, 13)

"""Error not implemented yet"""
# def test_error_on_wrong_shape():
#     model = CNNModel()
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         model(torch.randn((16, 3, 128)))
