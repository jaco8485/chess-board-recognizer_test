import torch
import hydra
from torch.utils.data import DataLoader
from data import ChessPositionsDataset
from model import CNNModel, ResNet
import torchvision.transforms as transforms
from loguru import logger
import time
import datetime
from utils import board_accuracy, draw_chessboard
from tqdm import tqdm
import timm
import wandb



def train(model, dataloader, optimizer, cfg):
    epochs = cfg.hyperparameters.epochs
    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    wandb.init(
        entity="rasmusmkn-danmarks-tekniske-universitet-dtu",
        project="chess-board-recognizer",
        config={"lr": cfg.hyperparameters.lr, "batch_size": cfg.hyperparameters.batch_size, "epochs": epochs},
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

            board_acc += board_accuracy(preds, truth, preds.shape[0]).item()

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        sample_image = inputs[0].unsqueeze(0)
        sample_label = labels[0]
        
        sample_truth = draw_chessboard(sample_label.argmax(dim=-1))
        sample_prediction = draw_chessboard(model(sample_image).argmax(dim=-1)[0])
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = board_acc / len(dataloader)
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}"
        )
        
        wandb.log({"train_loss": loss, "train_accuracy": epoch_accuracy})
        
        wandb.log({"images" : {"sample": wandb.Image(sample_image),"truth": wandb.Image(sample_truth),"prediction": wandb.Image(sample_prediction)} })


@hydra.main(version_base=None, config_path="../../configs", config_name="train.yaml")
def main(cfg):
    model = ResNet()

    if cfg.hyperparameters.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters())

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ]
    )

    chess_dataset = ChessPositionsDataset("data/", transform=transform)

    train_loader = DataLoader(
        chess_dataset, cfg.hyperparameters.batch_size, shuffle=True, num_workers=cfg.hyperparameters.num_workers, persistent_workers=True
    )

    train(model, train_loader, optimizer, cfg)

    torch.save(
        model.state_dict(),
        f"models/{cfg.hyperparameters.model}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M')}.pt",
    )


if __name__ == "__main__":
    main()
