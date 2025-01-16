import sys
from pathlib import Path
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import chess_board_recognizer.utils as utils
from chess_board_recognizer.data import ChessPositionsDataset
import matplotlib.pyplot as plt
import pytest
import torch



def test_train_dataset_lenght():
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    # Testing the train dataset
    dataset_train = ChessPositionsDataset("data", transform, "train")
    assert isinstance(dataset_train, Dataset)
    assert len(dataset_train) == 1000 #80000 Set to 1000 for testing purposes


def test_test_dataset_lenght():

    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_test = ChessPositionsDataset("data", transform, "test")
    assert isinstance(dataset_test, Dataset)
    assert len(dataset_test) == 1000 #20000 Set to 1000 for testing purposes

def test_train_data_shape():
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = ChessPositionsDataset("data", transform, "train")
    for x,y in dataset_train:
        assert x.shape == (3, 128, 128)
        assert y.shape == (8, 8, 13)

def test_test_data_shape():
    transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_test = ChessPositionsDataset("data", transform, "test")
    for x,y in dataset_test:
        assert x.shape == (3, 128, 128)
        assert y.shape == (8, 8, 13) 

