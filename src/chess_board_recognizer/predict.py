import torch
from model import CNNModel
import torchvision.transforms as transforms
from loguru import logger
from pathlib import Path
import typer
from utils import board_accuracy, draw_chessboard, per_piece_accuracy, from_fen_notation
from PIL import Image


def main(model_path: Path, image_path: Path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ]
    )

    image = transform(Image.open(image_path)).unsqueeze(0)

    truth = from_fen_notation(image_path.name.replace(".jpeg", "")).argmax(dim=-1)

    prediction = model(image).argmax(dim=-1).squeeze(0)

    logger.info(f"Board Accuracy: {board_accuracy(prediction, truth) * 100:.2f}%")

    piece_acc = per_piece_accuracy(prediction, truth)

    logger.info("Per piece accuracy:")

    for name, acc in piece_acc.items():
        logger.info(f"\t{name}: {acc.item() * 100:.2f}%")

    draw_chessboard(prediction, "chessboard.png").show()


if __name__ == "__main__":
    typer.run(main)
