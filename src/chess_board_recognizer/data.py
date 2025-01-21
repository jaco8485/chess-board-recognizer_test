from pathlib import Path
import typer
from torch.utils.data import Dataset
import zipfile
from loguru import logger
import glob
import os
from PIL import Image
from utils import from_fen_notation
import torchvision.transforms as transforms

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/koryakinp/chess-positions"


class ChessPositionsDataset(Dataset):
    """
    The chess position dataset consisting of 100000 images of online chess positions containing 3-15 chess pieces

    Will download the dataset to specified folder if it doesnt exist there.
    """

    def __init__(self, data_path: str = "data", transform: transforms.Compose = None, type: str = "train") -> None:
        self.data_path = Path(data_path)
        self.transform = transform

        if not self.__is_dataset_ready__():
            logger.info("Preparing ChessPositionsDataset")

            if not Path.exists(self.data_path / "data.zip"):
                logger.error(str(self.data_path / "data.zip") + " cannot be found")
                raise Exception(str(self.data_path / "data.zip") + " cannot be found")

            logger.info("Unpacking dataset")

            with zipfile.ZipFile(self.data_path / "data.zip", "r") as zip_ref:
                zip_ref.extractall(self.data_path)

            logger.info("Cleaning up")

            os.remove(self.data_path / "data.zip")

            logger.info("Finished downloading dataset")

        self.data_paths = glob.glob(str(self.data_path / type / "*"))

        self.data_paths = self.data_paths

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.data_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        fen_notation = from_fen_notation(Path(self.data_paths[index]).name.replace(".jpeg", ""))
        return image, fen_notation

    def __is_dataset_ready__(self):
        return Path.exists(self.data_path / "train") and Path.exists(self.data_path / "test")


def main(data_path: str):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    ChessPositionsDataset(data_path=data_path, transform=transform)


if __name__ == "__main__":
    typer.run(main)
