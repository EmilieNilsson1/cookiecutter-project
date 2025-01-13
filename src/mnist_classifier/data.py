import typer
import torch


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - images.mean()) / images.std()


def preprocess(raw_path: str, process_path: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_im, train_target = [], []
    for i in range(6):
        train_im.append(torch.load(f"{raw_path}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_path}/train_target_{i}.pt"))

    train_im = torch.cat(train_im, dim=0).unsqueeze(1).float()
    train_target = torch.cat(train_target, dim=0).long()

    test_im = torch.load(f"{raw_path}/test_images.pt").unsqueeze(1).float()
    test_target = torch.load(f"{raw_path}/test_target.pt").long()

    train_images = normalize(train_im)
    test_images = normalize(test_im)

    torch.save(train_images, f"{process_path}/train_images.pt")
    torch.save(train_target, f"{process_path}/train_target.pt")
    torch.save(test_images, f"{process_path}/test_images.pt")
    torch.save(test_target, f"{process_path}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("/Users/emilienilsson/Documents/DTU/9semester/02476MLOPS/cookiecutter-project/data/processed/train_images.pt")
    train_target = torch.load("/Users/emilienilsson/Documents/DTU/9semester/02476MLOPS/cookiecutter-project/data/processed/train_target.pt")
    test_images = torch.load("/Users/emilienilsson/Documents/DTU/9semester/02476MLOPS/cookiecutter-project/data/processed/test_images.pt")
    test_target = torch.load("/Users/emilienilsson/Documents/DTU/9semester/02476MLOPS/cookiecutter-project/data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
