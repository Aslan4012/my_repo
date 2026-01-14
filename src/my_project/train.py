import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pt
from pytorch_lightning.callbacks import ModelCheckpoint
import typer
from data import corrupt_mnist
from model import MyAwesomeModel


def train(lr: float = 1e-3, batch_size: int = 128, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    # Data
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model and trainer
    model = MyAwesomeModel()
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="models/",  # where to save
    #     filename="model-{epoch}",  # file name pattern
    #     save_top_k=1,  # keep best model
    #     monitor="train_loss",  # or val_loss if you have validation
    #     mode="min",
    # )

    trainer = pt.Trainer(
        max_epochs=epochs,
        accelerator="auto",  # uses CUDA/MPS/CPU automatically
        devices="auto",
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)
    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train()


def main() -> None:
    train()
