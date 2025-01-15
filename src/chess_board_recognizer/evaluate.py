import torch
from torch.utils.data import DataLoader
from data import ChessPositionsDataset
from model import CNNModel
import torchvision.transforms as transforms
from loguru import logger
from pathlib import Path
import typer
from utils import board_accuracy


def evaluate(model, dataloader):
    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("Starting evaluation")
    running_loss = 0
    board_acc = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        preds = outputs.argmax(dim=3)
        truth = labels.argmax(dim=3)

        board_acc += board_accuracy(preds, truth, preds.shape[0])

        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

    logger.info(
        f"Evaluation Metrics - Loss : {running_loss / len(dataloader)} | Average Accuracy : {board_acc / len(dataloader)}"
    )


def main(model_path: Path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    chess_dataset = ChessPositionsDataset("data/", type="test", transform=transform)

    test_loader = DataLoader(chess_dataset, 16, shuffle=False)

    evaluate(model, test_loader)


if __name__ == "__main__":
    typer.run(main)
