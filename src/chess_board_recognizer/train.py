import torch
import hydra
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ChessPositionsDataset
from model import CNNModel
import torchvision.transforms as transforms
from loguru import logger
import time
import datetime
from utils import board_accuracy

def train(model,dataloader,optimizer,cfg):
    epochs = cfg.hyperparameters.epochs
    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        board_acc = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            
            preds = outputs.argmax(dim=3)
            truth = labels.argmax(dim=3)
            
            board_acc += board_accuracy(preds,truth,preds.shape[0])
            
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {board_acc/ len(dataloader):.4f}")
    
@hydra.main(version_base=None,config_path="../../configs",config_name="train.yaml")
def main(cfg):
    
    
    model = CNNModel()
    
    if cfg.hyperparameters.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters())
        
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    chess_dataset = ChessPositionsDataset("data/",transform = transform)
    
    train_loader = DataLoader(chess_dataset,cfg.hyperparameters.batch_size,shuffle=True)

    train(model,train_loader,optimizer,cfg)
    
    torch.save(model.state_dict(), f"models/{cfg.hyperparameters.model}_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M')}.pt")
    
    
if __name__ == "__main__":
    main()