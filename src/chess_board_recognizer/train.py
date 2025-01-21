import torch
import hydra
from torch.utils.data import DataLoader
from data import ChessPositionsDataset
from model import CNNModel, ResNet
import torchvision.transforms as transforms
from loguru import logger
import time
import datetime
from utils import board_accuracy, draw_chessboard, per_piece_accuracy
from tqdm import tqdm
import wandb
import os


def train(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, cfg):
    epochs = cfg.hyperparameters.epochs
    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info("Starting training:")
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch Size: {cfg.hyperparameters.batch_size}")
    logger.info(f"Learning Rate: {cfg.hyperparameters.lr}")
    logger.info(f"Total Epochs: {cfg.hyperparameters.epochs}")
    logger.info(f"Optimizer: {type(optimizer).__name__}")

    wandb.init(
        entity="rasmusmkn-danmarks-tekniske-universitet-dtu",
        project="chess-board-recognizer",
        config={
            "lr": cfg.hyperparameters.lr,
            "batch_size": cfg.hyperparameters.batch_size,
            "epochs": cfg.hyperparameters.epochs,
        },
        name=f"{cfg.hyperparameters.model}-train",
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        board_acc = 0.0

        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            preds = outputs.argmax(dim=-1)
            truth = labels.argmax(dim=-1)

            board_acc += board_accuracy(preds, truth).item()

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        sample_image = inputs[0].unsqueeze(0)
        sample_label = labels[0].argmax(dim=-1)
        sample_prediction = model(sample_image).argmax(dim=-1)[0]

        sample_truth_board = draw_chessboard(sample_label)
        sample_prediction_board = draw_chessboard(model(sample_image).argmax(dim=-1)[0])

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = board_acc / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        wandb.log(
            {
                "sample": {
                    "image": wandb.Image(sample_image),
                    "truth": wandb.Image(sample_truth_board),
                    "prediction": wandb.Image(sample_prediction_board),
                    "board_accuracy": board_accuracy(sample_prediction, sample_label),
                    "per_piece_accuracy": per_piece_accuracy(sample_prediction, sample_label),
                },
                "train_loss": loss,
                "train_accuracy": epoch_accuracy,
            }
        )
    wandb.finish(0)


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment_cnn.yaml")
def main(cfg):
    # Loguru doesn't output to the hydra log file by default.
    logger.add(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "train.log"))

    if cfg.hyperparameters.model == "CNN":
        model = CNNModel()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
            ]
        )
    elif cfg.hyperparameters.model == "ResNet":
        model = ResNet()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ]
        )
    else:
        logger.error("Invalid network type in configuration. Available Models: CNN, ResNet")
        raise Exception("Specified network type is invalid")

    if cfg.hyperparameters.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif cfg.hyperparameters.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters())
    else:
        logger.warning("Invalid optimizer chosen, defaulting to adam")
        optimizer = torch.optim.Adam(model.parameters())

    chess_dataset = ChessPositionsDataset("data/", transform=transform)

    train_loader = DataLoader(chess_dataset, cfg.hyperparameters.batch_size, shuffle=True, num_workers=0)

    train(model, train_loader, optimizer, cfg)

    torch.save(
        model.state_dict(),
        f"models/{cfg.hyperparameters.model}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M')}.pt",
    )


if __name__ == "__main__":
    main()
