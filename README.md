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

# 📌 Project Checklist  

## ✅ Week 1  
### 🔹 Git & Environment Setup  
- [x] **Create a Git repository** (**M5**)  
- [x] **Ensure all team members have write access** (**M5**)  
- [x] **Create a dedicated environment to track dependencies** (**M2**)  

### 🔹 Project Structure & Code Initialization  
- [x] **Generate initial file structure using Cookiecutter** (**M6**)  
- [x] **Implement `data.py` to download and preprocess necessary data** (**M6**)  
- [ ] **Implement `model.py` with a basic model and `train.py` for training** (**M6**)  
- [ ] **Fill `requirements.txt` and `requirements_dev.txt` with dependencies** (**M2+M6**)  

### 🔹 Code Quality & Version Control  
- [ ] **Follow PEP8 coding standards** (**M7**)  
- [ ] **Add type hints and document essential parts of the code** (**M7**)  
- [ ] **Setup version control for data** (**M8**)  

### 🔹 CLI & Docker Setup  
- [ ] **Add CLI commands where applicable** (**M9**)  
- [ ] **Create Dockerfiles for your code** (**M10**)  
- [ ] **Build and test Dockerfiles locally** (**M10**)  

### 🔹 Configuration & Optimization  
- [ ] **Write configuration files for experiments** (**M11**)  
- [ ] **Use Hydra for managing hyperparameters** (**M11**)  
- [ ] **Profile code for optimization** (**M12**)  
- [ ] **Add logging for important events** (**M14**)  
- [ ] **Use Weights & Biases for experiment tracking** (**M14**)  
- [ ] **Consider running a hyperparameter optimization sweep** (**M14**)  
- [ ] **Use PyTorch Lightning to reduce boilerplate code** (**M15**)  

---

## ✅ Week 2  
### 🔹 Testing & CI/CD  
- [ ] **Write unit tests for data processing** (**M16**)  
- [ ] **Write unit tests for model construction/training** (**M16**)  
- [ ] **Measure code coverage** (**M16**)  
- [ ] **Setup continuous integration on GitHub** (**M17**)  
- [ ] **Add caching and multi-OS/Python/PyTorch testing** (**M17**)  
- [ ] **Add a linting step to CI pipeline** (**M17**)  
- [ ] **Setup pre-commit hooks for version control** (**M18**)  

### 🔹 Automated Workflows  
- [ ] **Create a workflow that triggers when data changes** (**M19**)  
- [ ] **Create a workflow for model registry changes** (**M19**)  

### 🔹 Cloud & Deployment Setup  
- [ ] **Store data in a GCP Bucket and integrate with data version control** (**M21**)  
- [ ] **Create a workflow for automatic Docker image builds** (**M21**)  
- [ ] **Train the model on GCP (Engine or Vertex AI)** (**M21**)  
- [ ] **Develop a FastAPI application for inference** (**M22**)  
- [ ] **Deploy the model in GCP (Cloud Functions or Cloud Run)** (**M23**)  

### 🔹 API & Performance Testing  
- [ ] **Write API tests and integrate them into CI/CD** (**M24**)  
- [ ] **Load test the API** (**M24**)  
- [ ] **Develop an ML deployment API using ONNX or BentoML** (**M25**)  
- [ ] **Create a frontend for the API** (**M26**)  

---

## ✅ Week 3  
### 🔹 Model Robustness & Drift Detection  
- [ ] **Evaluate model robustness against data drift** (**M27**)  
- [ ] **Deploy a drift detection API in the cloud** (**M27**)  

### 🔹 Monitoring & Alerting  
- [ ] **Instrument API with system metrics** (**M28**)  
- [ ] **Setup cloud monitoring for application** (**M28**)  
- [ ] **Implement alerting system in GCP** (**M28**)  

### 🔹 Performance Optimization  
- [ ] **Optimize data loading with distributed processing (if applicable)** (**M29**)  
- [ ] **Optimize model training with distributed training (if applicable)** (**M30**)  
- [ ] **Experiment with quantization, pruning, and model compilation for inference speedup** (**M31**)  

---

## 🎯 Extra Tasks  
- [ ] **Write project documentation** (**M32**)  
- [ ] **Publish documentation to GitHub Pages** (**M32**)  
- [ ] **Review project goals and outcomes**  
- [ ] **Create an architectural diagram of the MLOps pipeline**  
- [ ] **Ensure all team members understand all project components**  
- [ ] **Upload all final code to GitHub**  



