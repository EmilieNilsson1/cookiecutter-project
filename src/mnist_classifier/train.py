import os

import torch
import matplotlib.pyplot as plt
#import typer
#from omegaconf import OmegaConf
import hydra
import wandb
from pathlib import Path

from data import corrupt_mnist
from model import Classifier
#from mnist_classifier import corrupt_mnist, Classifier

#app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="../../configs", config_name="train_conf")
def train(config) -> None:
    """Train a model on MNIST data. Saves trained model to models/model.pth."""
    wandb.init(project="mnist_classifier", entity="emilienilsson-danmarks-tekniske-universitet-dtu")
    hparams = config.hyperparameters
    lr = hparams.lr
    batch_size = hparams.batch_size
    epochs = hparams.epochs
    torch.manual_seed(hparams.seed)

    print("Training start")
    print(f"Learning rate: {lr}, Batch size: {batch_size}, Epochs: {epochs}")

    # load model and data
    model = Classifier().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # using Adam optimizer and CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # save training loss and accuracy
    statistics = {"train_loss": [], "train_accuracy": []}
    for ep in range(epochs):
        model.train()
        for i, (im, target) in enumerate(train_dataloader):
            im, target = im.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(im)  # forward pass
            loss = criterion(output, target)
            loss.backward()  # backward pass
            optimizer.step()

            # save statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (output.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            # print statistics if epoch is divisible by 100
            if i % 100 == 0:
                print(f"Epoch {ep}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    

# def main():
#     typer.run(train)

if __name__ == "__main__":
   #main()
   train()
