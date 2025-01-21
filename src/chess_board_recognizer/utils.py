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


def from_fen_notation(fen_string: str) -> torch.Tensor:
    """
    Converts the given FEN string into a tensor of size 8x8x13 which represents the board state.

    Args:
        fen_string (str): The FEN string representing the cheesboard.

    Returns:
        torch.Tensor: The output is a one hot encoded 8x8x13 tensor representing the chessboard as given by the FEN string
    """
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


def board_accuracy(board_one: torch.Tensor, board_two: torch.Tensor) -> float:
    """
    Calculates how many squares of 2 boards are equal and returns the fraction of how many squares are equal over amount of total squares

    Boards must not be one hot encoded

    Args:
        board_one (torch.Tensor): The first board as an 8x8 or Bx8x8 tensor
        board_two (torch.Tensor): The second board as an 8x8 or Bx8x8 tensor

    Returns:
        float: The accuracy of how many squares are the same in both boards
    """
    
    if board_one.shape != board_two.shape:
        raise ValueError(f"Both boards must have the same shape. board_one shape: {board_one.shape}, board_two shape: {board_two.shape}")

    if board_one.ndim != 2 and board_one.ndim != 3:
        raise ValueError(f"Expected board dimensionality to be 2 or 3. Actual dim: {board_one.ndim}")
    
    if board_one.shape[-2:] != (8,8):
        raise ValueError(f"Boards are not of size 8x8. Actual size {board_one.shape}")
    
    batch_size = 1
    
    if board_one.ndim == 3:
        batch_size = board_one.shape[0]
    
    return (board_one == board_two).sum() / (batch_size * 64)


def per_piece_accuracy(predicted_board: torch.Tensor, true_board: torch.Tensor) -> dict:
    """
    Calculates the accuracy per piece.
    The accuracy is calculated by comparing how many pieces the predicted board has in common with the ground truth board divided by the largest amount of that piece found on any board.
    
    E.g.
    
    Predicted board has predicted 2 white pawns in the correct position and 1 in the incorrect position, while the ground truth only has 2 white pawns. In this case the accuracy would be 0.66

    Args:
        predicted_board (torch.Tensor): The board that has been predicted as an 8x8 or Bx8x8 tensor. Supports values 0-12
        true_board (torch.Tensor): The ground truth board as an 8x8 or Bx8x8 tensor. Supports values 0-12

    Returns:
        dict: A dictionary that contains how many pieces the predicted board has in common with the ground turth board.
    """
    
    if predicted_board.shape != true_board.shape:
        raise ValueError(f"Both boards must have the same shape. predicted_board shape: {predicted_board.shape}, true_board shape: {true_board.shape}")
    
    if predicted_board.ndim != 2 and predicted_board.ndim != 3:
        raise ValueError(f"Expected board dimensionality to be 2 or 3. Actual dim: {predicted_board.ndim}")
    
    if predicted_board.shape[-2:] != (8,8):
        raise ValueError(f"Boards are not of size 8x8. Actual size {predicted_board.shape}")
    
    batch_size = 1
    
    if predicted_board.ndim == 3:
        batch_size = predicted_board.shape[0]
    
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


def draw_chessboard(board: torch.Tensor, save_path: str = "") -> Image:
    """
    
    Draws the given board to a PIL Image and optionally saves it to a path. Uses the pieces from the styles folder
    
    Only values from 0-12 are accepted.

    Args:
        board (torch.Tensor): The board to be drawn as an 8x8 tensor
        save_path (str, optional): The path to save the image to. Defaults to "".

    Returns:
        Image: The board in visual format.
    """
    
    if board.shape != (8,8):
        raise ValueError("Board is not of size 8x8")
    
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

    if save_path != "":
        board_image.save(save_path)
        
    return board_image
