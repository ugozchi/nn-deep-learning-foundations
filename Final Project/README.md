# Final project: an intelligent museum guide

We are the AI team of a (fictional) mid-sized French fine-arts museum. The Museum wants a guide, on visitors' phones and on gallery kiosks, that:

1. **Recognises what the visitor is looking at**: from a photo of a painting, predict its style (Impressionism, Baroque, Cubism…) or its genre (portrait, landscape, still life…).
2. **Answers questions in natural language**: "Which painting shows a stormy sea?" returns the best-matching artworks of the collection.

The director also wants a clear recommendation: the kiosks run on modest CPUs, the team is small and the budget is limited. **Is a neural network worth it in production, or would a simpler classical model do the job?**

Full brief: [`consign.md`](consign.md). Deadline: **30 October 2026** (Git repository + 10 min presentation + 5 min Q&A).

## Deliverables

- [x] A feed-forward network in PyTorch for artwork classification: [`notebooks/mlp.ipynb`](notebooks/mlp.ipynb)
- [x] Training monitoring and failure diagnosis: [`notebooks/monitoring.ipynb`](notebooks/monitoring.ipynb), [`notebooks/diagnosis.ipynb`](notebooks/diagnosis.ipynb)
- [x] A pretrained CNN (ResNet) for image classification: [`notebooks/resnet.ipynb`](notebooks/resnet.ipynb)
- [ ] A recommendation to the director, backed by measurements (accuracy, CPU latency, model size)
- [ ] The 10-minute presentation

Work is tracked in the GitHub issues and milestones.

## Layout

```
Final Project/
├── consign.md               project brief
├── make_wikiart_subset.py   builds the WikiArt subset (provided)
├── requirements.txt
├── src/                     shared code
│   ├── data.py              WikiArt subset, shared split, DataLoaders, flip augmentation
│   ├── train.py             training loop, early stopping, metrics, json saving
│   ├── models.py            MLP and pretrained ResNet-18
│   └── plots.py             learning curves and run comparisons
├── notebooks/               experiments, to run in order (see below)
├── results/                 metrics (json/csv) and figures; model weights (*.pt) are git-ignored
├── presentation/            slides for the oral defence
└── data/                    WikiArt subset, local only (git-ignored)
```

## Data

The data comes from the `huggan/wikiart` dataset on the Hugging Face Hub, through `make_wikiart_subset.py`. **WikiArt images must not be committed**: we commit the script, the exact command and `subset_info.json`.

We classify paintings by **style**, with five visually contrasted movements: **Impressionism, Baroque, Cubism, Ukiyo-e and Pop Art**, 300 images each, 128 × 128 px. Style is what the guide needs to tell the story of a movement. These five differ in palette, brushwork and composition, so the task can be learnt from 1 500 images, and there is still room for a gap between a classical model and a CNN.

Build it from `Final Project/` (about 4.2 GB streamed, up to about 1h30 on a home connection, re-running resumes):

```bash
pip install -r requirements.txt
python make_wikiart_subset.py --label style \
    --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
    --per-class 300 --size 128 --out data/wikiart_style5
```

The seed and the dataset revision are pinned, so everybody gets the same images. Check that your `data/wikiart_style5/subset_info.json` has the same `counts` and `shards` as the committed one.

Class indices follow the alphabetical folder order of `ImageFolder`, not the order of the command: Baroque 0, Cubism 1, Impressionism 2, Pop_Art 3, Ukiyo_e 4 (the `classes` list in `subset_info.json`).

## Reproduce the results

Set up the environment of the [root README](../README.md), then `pip install -r requirements.txt` here and build the data (above). Run the notebooks **in this order**, because each one reads what the previous ones saved in `results/`:

| # | Notebook | What it does | Writes to `results/` | Time |
| --- | --- | --- | --- | --- |
| 1 | [`baseline`](notebooks/baseline.ipynb) | majority class, logistic regressions on pixels and on colour histograms | `baseline.json`, `baseline_confusion.png` | 15 s |
| 2 | [`mlp`](notebooks/mlp.ipynb) | feed-forward network: architecture, activation, regularisation | `mlp.json`, `mlp_*.png`, `mlp.pt` | 35 s |
| 3 | [`resnet`](notebooks/resnet.ipynb) | pretrained ResNet-18: head only, fine-tuning, features + logistic regression | `resnet.json`, `resnet_*.png`, `resnet.pt` | 1.5 min on GPU |
| 4 | [`monitoring`](notebooks/monitoring.ipynb) | learning rate, gradient norm per layer, activations | `monitoring.json`, `monitoring_*.png` | 10 s |
| 5 | [`diagnosis`](notebooks/diagnosis.ipynb) | each failure: symptom, diagnosis, fix, measured effect | `diagnosis.csv`, `diagnosis_*.png` | 20 s |

Times measured on an Apple M4 Pro laptop. In Jupyter, use *Restart & Run All*; from the command line:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 baseline.ipynb
```

- `resnet` trains on the first device available: a CUDA GPU, the Apple GPU (MPS), else the CPU, where one epoch is about 8 times slower. Its first run downloads the ImageNet weights of ResNet-18 (45 MB) into the PyTorch cache. The other notebooks run on the CPU.
- The weights `mlp.pt` and `resnet.pt` are git-ignored. `diagnosis` loads `resnet.pt`, so run `resnet` first.
- Every notebook fixes its seeds: running one twice on the same machine gives the same numbers (checked for each of them). On another device, the ResNet numbers may differ slightly.

## Results

Test set: 225 paintings, 45 per style, the same for every model.

| Model | Accuracy | Macro-F1 | Source |
| --- | --- | --- | --- |
| Majority class | 0.200 | 0.067 | `results/baseline.json` |
| Logistic regression, raw pixels (16 px) | 0.369 | 0.339 | `results/baseline.json` |
| Logistic regression, colour histograms | 0.396 | 0.380 | `results/baseline.json` |
| MLP, 3 × 512, dropout and flips | 0.431 | 0.423 | `results/mlp.json` |
| Logistic regression on ResNet-18 features | 0.769 | 0.769 | `results/resnet.json` |
| ResNet-18 fine-tuned | 0.796 | 0.792 | `results/resnet.json` |

With 225 test paintings, the standard error of an accuracy is about 3 points. The failures met on the way, and how each fix was measured, are summarised in `results/diagnosis.csv`.
