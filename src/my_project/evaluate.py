import torch
import typer
import pytorch_lightning as pl
from src.my_project.data import corrupt_mnist
from src.my_project.model import MyAwesomeModel


def evaluate(model_checkpoint: str) -> None:
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    # Load model from checkpoint
    model = MyAwesomeModel.load_from_checkpoint(model_checkpoint)

    # Data
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,   # no logging needed here
    )

    # Run validation
    results = trainer.validate(model, dataloaders=test_dataloader)

    print(results)


if __name__ == "__main__":
    typer.run(evaluate)