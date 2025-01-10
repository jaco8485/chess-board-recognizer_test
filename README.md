# chess_board_recognizer

## Project Description


The overall goal of the project is to create a model that can take images of online chess boards, detect the position of the pieces and provide a FEN notation containing the piece configurations. While FEN is able to capture information of how the pieces moved during the game, we will not be including this in our model, as the moves you use to get to that position are almost always ambiguous.


We plan to use the [PyTorch image models](https://github.com/huggingface/pytorch-image-models) framework to define our image models. This will provide us with different options of models to choose from. As the project is available as a pip package we will include it in the project by adding it to the requirements.txt file.


We will use the “[Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions)” dataset from Kaggle, which contains 100,000 images of chess boards with 5-15 pieces in random configurations. All images are 400x400 pixels and were generated using 28 styles of chess boards and 32 styles of chess pieces totaling 896 board/piece style combinations. All images were generated using a [custom-build tool](https://github.com/koryakinp/chess-generator), so we can generate more if needed. Due to the way the images are generated, some positions may be illegal such as both kings are under check.

The dataset is divided into two subsets:

- Training set contains 80,000 images

- Test set contains 20,000 images

Labels are in a filename in FEN (Forsyth–Edwards Notation) format, but with dashes instead of slashes.

The full data set is about 2.3gb in size.


We expect to use some kind of CNN, the pytorch image model framework provides quite a few different models to choose from so we haven’t settled on a specific one. We’ll likely start with a simple CNN that we define ourselves in pytorch to do some initial benchmarking and then later switch to a model from the framework.



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
