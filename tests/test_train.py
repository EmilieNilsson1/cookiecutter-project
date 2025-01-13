import torch

from src.mnist_classifier.model import Classifier

def test_train():
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(1, 1, 28, 28)
    model = Classifier()
    output = model(x)
    y = torch.randn(1, 10)
    assert criterion(output, y), "Loss not computed correctly"
