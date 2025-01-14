from pathlib import Path
from urllib.request import urlretrieve
import typer
from torch.utils.data import Dataset
import zipfile
import shutil
import logging
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt


class ChessPositionsDataset(Dataset):
    """
    The chess position dataset consisting of 100000 images of online chess positions containing 3-15 chess pieces
    
    Will download the dataset to specified folder if it doesnt exist there.
    """

    def __init__(self, raw_data_path: Path, type : str = "train") -> None:
        self.data_path = raw_data_path        

        DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/koryakinp/chess-positions"

        if not Path.exists(raw_data_path):
            logging.getLogger().log(logging.DEBUG,"Dataset not found, downloading it instead.")

            Path.mkdir(raw_data_path,parents=True)

            urlretrieve(DATASET_URL,raw_data_path / "data.zip")

            with zipfile.ZipFile(raw_data_path / "data.zip", 'r') as zip_ref:
                zip_ref.extractall(raw_data_path)

            # The dataset is contained "twiceish" and is being cleaned up.
            os.remove(raw_data_path / "data.zip")
            shutil.rmtree(raw_data_path / "train")
            shutil.rmtree(raw_data_path / "test")

            shutil.move((raw_data_path/"dataset/train"),(raw_data_path / "train"))
            shutil.move((raw_data_path/"dataset/test"),(raw_data_path / "test"))
            shutil.rmtree((raw_data_path/"dataset"))

        self.data_paths = glob.glob(str(raw_data_path/ type / "*"))
        
    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int):
        return (Image.open(self.data_paths[index]),Path(self.data_paths[index]).name.replace(".jpeg",""))


def main(data_path : Path):
    dataset = ChessPositionsDataset(data_path)

    img, label = dataset.__getitem__(0)

    print(img)
    print(label)

if __name__ == "__main__":
    typer.run(main)
