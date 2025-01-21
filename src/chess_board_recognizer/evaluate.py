import torch
from torch.utils.data import DataLoader
from data import ChessPositionsDataset
from model import CNNModel, ResNet
import torchvision.transforms as transforms
from loguru import logger
import typer
from utils import board_accuracy
from tqdm import tqdm


def evaluate(model, dataloader):
    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("Starting evaluation")
    running_loss = 0
    board_acc = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        preds = outputs.argmax(dim=3)
        truth = labels.argmax(dim=3)

        board_acc += board_accuracy(preds, truth)

        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

    logger.info(
        f"Evaluation Metrics - Loss : {running_loss / len(dataloader)} | Average Accuracy : {board_acc / len(dataloader)}"
    )


def main(model_path: str):
    model_name = model_path.split("_")[0]

    if "CNN" in model_name:
        model = CNNModel()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
            ]
        )
    elif "ResNet" in model_name:
        model = ResNet()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ]
        )
    else:
        raise TypeError("Model typer not found")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    chess_dataset = ChessPositionsDataset("data/", type="test", transform=transform)

    test_loader = DataLoader(chess_dataset, 16, shuffle=False, num_workers=8)

    evaluate(model, test_loader)


if __name__ == "__main__":
    typer.run(main)
