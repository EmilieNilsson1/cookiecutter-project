import os

import pytest
from torch.utils.data import Dataset

from mnist_classifier.data import corrupt_mnist


@pytest.mark.skipif(not os.path.exists("data/processed/test_images.pt"), reason="Data files not found")
def test_data():
    dataset_train, dataset_test = corrupt_mnist()
    assert len(dataset_train) == 30000, "Dataset size should be 30000"
    assert len(dataset_test) == 5000, "Dataset size should be 5000"
    for dataset in [dataset_train, dataset_test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Image should be 1x28x28"
            assert y in range(10), "Label should be in range 0-9"
