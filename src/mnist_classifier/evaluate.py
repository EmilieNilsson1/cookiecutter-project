import torch
import typer

from mnist_classifier import Classifier, corrupt_mnist

app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model. Print test accuracy."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # load model
    model = Classifier().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    # data
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)  # forward pass
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()  # accuracy
            total += target.size(0)
    print(f"Test accuracy: {correct / total}")


def main():
    typer.run(evaluate)


if __name__ == "__main__":
    main()
