# Neural Networks & Deep Learning Foundations

Coursework for **MSA-DATB09-03 · Neural Networks & Deep Learning Foundations** (M2): the eight lab sessions and the final project.

## Contents

| Folder | Topic |
| --- | --- |
| [`Session 1`](Session%201) | TP01: the perceptron in NumPy |
| [`Session 2`](Session%202) | TP02: logistic regression, from NumPy to PyTorch |
| [`Session 3`](Session%203) | TP03: a first neural network |
| [`Session 4`](Session%204) | TP04: PyTorch in depth, the complete training pipeline |
| [`Session 5`](Session%205) | TP05: a vision project, activations and initialisation |
| [`Session 6`](Session%206) | TP06: training better networks |
| [`Session 7`](Session%207) | TP07: convolutional neural networks |
| [`Session 8`](Session%208) | TP08: recurrent networks, LSTM and attention |
| [`Final Project`](Final%20Project) | Museum guide: artwork classification with an MLP and a pretrained ResNet |

Each session folder holds the course slides (`course.pdf`) and the lab notebook(s).

## Setup

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The final project needs a few extra packages, listed in [`Final Project/requirements.txt`](Final%20Project/requirements.txt).

Datasets are downloaded by the notebooks or scripts and are never committed (see `.gitignore`).

## Team workflow

The final project is developed through issues and pull requests: see [CONTRIBUTING.md](CONTRIBUTING.md).
