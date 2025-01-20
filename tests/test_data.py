from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.chess_board_recognizer.data import ChessPositionsDataset


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

dataset_train = ChessPositionsDataset("data", transform, "train")
dataset_test = ChessPositionsDataset("data", transform, "test")

def test_train_dataset_lenght():
    assert isinstance(dataset_train, Dataset)
    assert len(dataset_train) == 80000

def test_test_dataset_lenght():
    assert isinstance(dataset_test, Dataset)
    assert len(dataset_test) == 20000 

def test_train_data_shape():
    for x,y in dataset_train:
        assert x.shape == (3, 128, 128)
        assert y.shape == (8, 8, 13)

def test_test_data_shape():
    for x,y in dataset_test:
        assert x.shape == (3, 128, 128)
        assert y.shape == (8, 8, 13) 

def test_train_nummber_of_pieces():
    for _,y in dataset_train:
        assert y[:,:,1:].sum() <= 15 and y[:,:,1:].sum() >= 5


def test_test_nummber_of_pieces():
    for _,y in dataset_test:
        assert y[:,:,1:].sum() <= 15 and y[:,:,1:].sum() >= 5

def test_train_kings():
    for _,y in dataset_train:
        assert y[:,:,6].sum() == 1 and y[:,:,12].sum() == 1

def test_test_kings():
    for _,y in dataset_test:
        assert y[:,:,6].sum() == 1 and y[:,:,12].sum() == 1

def test_train_one_pices_per_square():
    for _,y in dataset_train:
        assert (y[:,:,1:] > 1).sum() == 0

def test_test_one_pices_per_square():
    for _,y in dataset_test:
        assert (y[:,:,1:] > 1).sum() == 0