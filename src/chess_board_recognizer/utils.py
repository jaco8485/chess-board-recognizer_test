import torch
import torch.nn.functional as f
from PIL import Image, ImageDraw

piece = {"P": 1, "R": 2, "B": 3, "N": 4, "Q": 5, "K": 6, "p": 7, "r": 8, "b": 9, "n": 10, "q": 11, "k": 12}
piece_description = {
    0: "Blank Space",
    1: "White Pawn",
    2: "White Rook",
    3: "White Bishop",
    4: "White Knight",
    5: "White Queen",
    6: "White King",
    7: "Black Pawn",
    8: "Black Rook",
    9: "Black Bishop",
    10: "Black Knight",
    11: "Black Queen",
    12: "Black King",
}


def from_fen_notation(fen_string: str):
    board = torch.zeros((8, 8, 13))
    fen_rows = fen_string.split("-")

    for i, fen_row in enumerate(fen_rows):
        n = 0
        for char in fen_row:
            if str.isdigit(char):
                for j in range(int(char)):
                    board[i, n + j] = f.one_hot(torch.tensor(0), 13)
                n += int(char)
            else:
                board[i, n] = f.one_hot(torch.tensor(piece[char]), 13)
                n += 1

    return board


def board_accuracy(board_one: torch.Tensor, board_two: torch.Tensor, batch_size: int) -> float:
    """
    Calculates how many squares of 2 boards are equal and returns the fraction of how many squares are equal over amount of total squares

    Boards must not be one hot encoded
    """
    return (board_one == board_two).sum() / (batch_size * 64)


def per_piece_accuracy(predicted_board: torch.Tensor, true_board: torch.Tensor, batch_size: int) -> dict:
    accuracy_table = {}
    for i in range(13):
        if (true_board == i).sum() != 0:
            accuracy_table[piece_description[i]] = torch.logical_and(
                (predicted_board == i), (true_board == i)
            ).sum() / (max((true_board == i).sum(), (predicted_board == i).sum()) * batch_size)
        else:
            if (predicted_board == i).sum() == 0:
                accuracy_table[piece_description[i]] = torch.tensor(1.0)
            else:
                accuracy_table[piece_description[i]] = torch.tensor(0.0)
    return accuracy_table


def draw_chessboard(board: torch.Tensor, save_path: str):
    chessboard = board
    square_size = 50
    board_size = square_size * 8

    board_image = Image.new("RGB", (board_size, board_size), "white")
    draw = ImageDraw.Draw(board_image)

    for row in range(8):
        for col in range(8):
            color = "white" if (row + col) % 2 == 0 else "gray"
            draw.rectangle(
                [col * square_size, row * square_size, (col + 1) * square_size, (row + 1) * square_size], fill=color
            )

    piece_images = {}
    for value, name in piece_description.items():
        if name != "Blank Space":
            piece_images[value] = Image.open(f"styles/{value}.png").resize((square_size, square_size)).convert("RGBA")

    for row in range(8):
        for col in range(8):
            piece = chessboard[row][col].item()
            if piece != 0:
                board_image.paste(piece_images[piece], (col * square_size, row * square_size), piece_images[piece])

    board_image.save(save_path)
    board_image.show()
