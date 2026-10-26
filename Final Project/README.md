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
