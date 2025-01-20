import pytest
import torch
import utils as utils


def test_from_fen_notation_full_borad():
    fen = "1b1B1b2-2pK2q1-4p1rB-7k-8-8-3B4-3rb3"
    board = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], 
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], 
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], 
                            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], 
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    assert torch.allclose(utils.from_fen_notation(fen), board)
    
def test_from_fen_notation_empty_board():
    fen = "8-8-8-8-8-8-8-8"
    board = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    assert torch.allclose(utils.from_fen_notation(fen), board)

def test_from_fen_notation_one_piece():
    fen = "8-8-8-8-8-8-8-1k"
    borad = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    assert torch.allclose(utils.from_fen_notation(fen), borad)

def test_from_fen_notation_part_board():
    fen = "2pK2q1-4p1rB"
    borad = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    assert torch.allclose(utils.from_fen_notation(fen), borad)

def test_from_fen_notation_invalid_fen():
    with pytest.raises(KeyError):
        utils.from_fen_notation("invalid fen")

def test_from_fen_notation_empty_fen():
    fen = ""
    board = torch.zeros((8, 8, 13))
    assert torch.allclose(utils.from_fen_notation(fen), board)

def test_board_accuracy_empty_borads():
    board_one = torch.zeros((8, 8, 13))
    board_two = torch.zeros((8, 8, 13))
    batch_size = 1
    assert utils.board_accuracy(torch.argmax(board_one, dim=-1), torch.argmax(board_two, dim=-1), batch_size) == 1

# def test_board_accuracy_empty_borads_one_hot():
#     board_one = torch.zeros((8, 8, 13))
#     board_two = torch.zeros((8, 8, 13))
#     batch_size = 1
#     assert utils.board_accuracy(board_one, board_two, batch_size) == 1

def test_board_accuracy_diffrent_boards():
    board_one = utils.from_fen_notation("1b1B1b2-2pK2q1-4p1rB-7k-8-8-3B4-3rb3")
    board_two = utils.from_fen_notation("1b1B2n1-3K1PN1-B6N-3k1B2-5p2-8-1B3q2-r1b5")
    batch_size = 1
    assert utils.board_accuracy(torch.argmax(board_one,dim=-1), torch.argmax(board_two,dim=-1), batch_size) == 0.6875

def test_board_accuracy_invalid_borads():
    board_one = torch.zeros((8, 8, 13))
    board_two = torch.ones((9, 9, 13))
    batch_size = 1
    with pytest.raises(RuntimeError):
        utils.board_accuracy(torch.argmax(board_one,dim=-1), torch.argmax(board_two,dim=-1), batch_size)
