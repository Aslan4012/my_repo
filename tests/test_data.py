from torch.utils.data import Dataset
from my_project.data import corrupt_mnist


def test_corrupt_mnist():
    train_set, test_set = corrupt_mnist()
    assert isinstance(train_set, Dataset)
    assert isinstance(test_set, Dataset)