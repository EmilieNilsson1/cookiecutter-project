import importlib

import pytest
import torch

import mnist_classifier.model
from mnist_classifier.model import Classifier

importlib.reload(mnist_classifier.model)
print(mnist_classifier.model.__file__)


def test_model():
    x = torch.randn(1, 1, 28, 28)
    model = mnist_classifier.model.Classifier()
    output = model(x)
    assert output.shape == (1, 10), "output shape should be 1x10"


def test_dimension():
    model = mnist_classifier.model.Classifier()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    # with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
    #     model(torch.randn(1,1,27,29))


if __name__ == "__main__":
    # test_model()
    # test_dimension()
    model = mnist_classifier.model.Classifier()
    x = torch.randn(1, 1, 27, 28)
    model(x)
