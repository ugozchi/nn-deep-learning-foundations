# Final project: an intelligent museum guide

We are the AI team of a (fictional) mid-sized French fine-arts museum. The Museum wants a guide, on visitors' phones and on gallery kiosks, that:

1. **Recognises what the visitor is looking at**: from a photo of a painting, predict its style (Impressionism, Baroque, Cubism…) or its genre (portrait, landscape, still life…).
2. **Answers questions in natural language**: "Which painting shows a stormy sea?" returns the best-matching artworks of the collection.

The director also wants a clear recommendation: the kiosks run on modest CPUs, the team is small and the budget is limited. **Is a neural network worth it in production, or would a simpler classical model do the job?**

Full brief: [`consign.txt`](consign.txt). Deadline: **30 October 2026** (Git repository + 10 min presentation + 5 min Q&A).

## Deliverables

- [ ] A feed-forward network in PyTorch for artwork classification
- [ ] Training monitoring and failure diagnosis
- [ ] A pretrained CNN (ResNet) for image classification
- [ ] A recommendation to the director, backed by measurements (accuracy, CPU latency, model size)
- [ ] The 10-minute presentation

Work is tracked in the GitHub issues and milestones.

## Layout

```
Final Project/
├── consign.txt              project brief
├── make_wikiart_subset.py   builds the WikiArt subset (provided)
├── requirements.txt
├── src/                     shared code: data loading, models, training loop, metrics
├── notebooks/               experiments (one owner per notebook)
├── results/                 metrics (json/csv) and figures kept as evidence
├── presentation/            slides for the oral defence
└── data/                    WikiArt subset, local only (git-ignored)
```

## Data

The data comes from the `huggan/wikiart` dataset on the Hugging Face Hub, through `make_wikiart_subset.py`. **WikiArt images must not be committed**: we commit the script, the exact command and `subset_info.json`.

```bash
pip install -r requirements.txt

# 1. Inspect the classes
python make_wikiart_subset.py --list-classes --label style

# 2. Check the plan (no download)
python make_wikiart_subset.py --label style --classes <CLASSES> --per-class 300 --size 128 --out data/<NAME> --dry-run

# 3. Build the subset (same command without --dry-run)
```

The exact command we use will be recorded here once the classes are chosen.
