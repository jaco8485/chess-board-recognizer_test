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
from chess_board_recognizer.utils import from_fen_notation
from torchvision.transforms import Compose

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/koryakinp/chess-positions"

class ChessPositionsDataset(Dataset):
    """
    The chess position dataset consisting of 100000 images of online chess positions containing 3-15 chess pieces

    Will download the dataset to specified folder if it doesnt exist there.
    """

    def __init__(self, raw_data_path: str, transform: Compose, type: str = "train") -> None:
        self.data_path = Path(raw_data_path)
        self.transform = transform


        if not Path.exists(self.data_path):
            logging.getLogger(__name__).log(logging.INFO, "Dataset not found, downloading it instead.")

            Path.mkdir(self.data_path, parents=True)

            urlretrieve(DATASET_URL, self.data_path / "data.zip")

            logging.getLogger(__name__).log(logging.INFO, "Upacking data.zip")

            with zipfile.ZipFile(self.data_path / "data.zip", "r") as zip_ref:
                zip_ref.extractall(self.data_path)

            logging.getLogger(__name__).log(logging.INFO, "Cleaning up")

            # The dataset is contained "twiceish" and is being cleaned up.
            os.remove(self.data_path / "data.zip")
            shutil.rmtree(self.data_path / "train")
            shutil.rmtree(self.data_path / "test")

            shutil.move((self.data_path / "dataset/train"), (self.data_path / "train"))
            shutil.move((self.data_path / "dataset/test"), (self.data_path / "test"))
            shutil.rmtree((self.data_path / "dataset"))

            logging.getLogger(__name__).log(logging.INFO, "Finished downloading dataset")

        self.data_paths = glob.glob(str(self.data_path / type / "*"))[:1000]

        self.data = [self.transform(Image.open(path)) for path in self.data_paths]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image = self.data[index]
        fen_notation = from_fen_notation(Path(self.data_paths[index]).name.replace(".jpeg", ""))
        return image, fen_notation


def main(data_path: Path):
    dataset = ChessPositionsDataset(data_path)

    img, label = dataset.__getitem__(0)

    print(img)
    print(label)


if __name__ == "__main__":
    typer.run(main)
