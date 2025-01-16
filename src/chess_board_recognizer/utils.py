import torch
import torch.nn.functional as f

piece = {"P": 1, "R": 2, "B": 3, "N": 4, "Q": 5, "K": 6, "p": 7, "r": 8, "b": 9, "n": 10, "q": 11, "k": 12}


def from_fen_notation(fen_string: str):
    board = torch.zeros((8, 8, 13))

    fen_rows = fen_string.split("-")

    for i, fen_row in enumerate(fen_rows):
        n = 0
        for char in fen_row:
            if str.isdigit(char):
                n += int(char)
            else:
                board[i, n] = f.one_hot(torch.tensor(piece[char]), 13)

    return board


def board_accuracy(board_one: torch.Tensor, board_two: torch.Tensor, batch_size: int) -> float:
    """
    Calculates how many squares of 2 boards are equal and returns the fraction of how many squares are equal over amount of total squares

    Boards must not be one hot encoded
    """
    return (board_one == board_two).sum() / (batch_size * 64)
