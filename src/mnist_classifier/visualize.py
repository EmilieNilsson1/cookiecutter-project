import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mnist_classifier import Classifier, corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    model = Classifier().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # remove final layer
    model.fc = torch.nn.Identity()

    _, test_data = corrupt_mnist()
    test_dataset = torch.utils.data.DataLoader(test_data, batch_size=32)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in test_dataset:
            images, target = batch
            images, target = images.to(DEVICE), target.to(DEVICE)
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig("reports/figures/embeddings.png")


if __name__ == "__main__":
    typer.run(evaluate)
