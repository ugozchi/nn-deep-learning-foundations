# Oral preparation

Slides: [`museum_guide.pptx`](museum_guide.pptx), 11 slides for 10 minutes, with speaker notes on every slide. Every number comes from `results/`.

## Speakers and timing (proposal)

| Part | Slides | Topic | Speaker | Time |
| --- | --- | --- | --- | --- |
| 1 | 1–4 | The question, the data, the classical baselines | Fefe | 3 min |
| 2 | 5–7 | The MLP, monitoring, diagnosis | Ugo | 3 min 30 |
| 3 | 8–11 | The ResNet, the errors, the cost, the recommendation | Thomas | 3 min 30 |

Slides 10 and 11 are completed with the CPU benchmark (#9) and the recommendation (#10). Two timed rehearsals before the 30th.

## Likely questions

**Data and protocol**

- *Why these five styles?* They differ visibly in palette, brushwork and composition, so 300 paintings per style are enough to learn, and they leave room for a gap between a classical model and a CNN. Style is also what the guide needs to tell the story of a movement.
- *How did you avoid overfitting the test set?*
  - Every choice (image size, `C`, architecture, regularisation, fine-tuning variant) was made on the validation set.
  - The test set was used once per final model.
  - The normalisation statistics come from the training images only.
- *Is 0.796 really better than 0.769?* No. On 225 test paintings an accuracy has a standard error of about 3 points, so a 2.7-point gap is within the noise. 0.43 → 0.80, on the other hand, is far beyond it.
- *Why 3 seeds?* One run on 225 validation paintings can move by a few points with the seed alone (initial weights, batch order). The mean and the standard deviation tell whether a difference is real.
- *Why macro-F1?* It is the mean of the F1 scores of the five styles, so every style counts equally. Our classes are balanced, so it stays close to the accuracy, but it would reveal a style that is ignored if the collection became unbalanced.

**The MLP**

- *Why is the MLP so weak?* It has one weight per pixel. The same brushwork or shape at another place in the painting uses other weights, and the network has no notion of local texture. Regularisation shrinks the train–validation gap (0.63 → 0.08) but not the ceiling (about 0.40). Its per-class profile is the same as that of the colour models.
- *Why 32 × 32 pixels?* On raw pixels, 16, 32 and 64 pixels gave the same accuracy (baseline notebook). 32 keeps some detail with 3 072 inputs, so the first layer has 1.6 million weights instead of 6.3 million at 64 pixels.
- *How do you know it is overfitting?*
  - The training accuracy reaches 0.99 while the validation accuracy stays at 0.36.
  - The validation loss is lowest at epoch 4, then rises to 3.9 while the training loss keeps falling.
- *Why does the sigmoid learn more slowly?* Its derivative is at most 0.25, and 30 % of the first-layer outputs are saturated, where the derivative is almost 0. Backpropagation multiplies these derivatives, so the first layer gets a gradient 68 times smaller than the output layer. The result: best epoch 18 instead of 4.
- *Why Adam with lr 1e-4?* At 1e-2 the loss explodes at the first epoch (10.3), then oscillates. At 1e-6 it barely moves in 30 epochs. 1e-4 trains steadily (monitoring notebook).

**The ResNet**

- *Why keep the batch norm in `eval()` mode?* With 1 050 paintings and batches of 64, the batch statistics are noisy. As in TP07, we keep the statistics learned on ImageNet. ResNet-18 has no dropout, so `eval()` changes nothing else, and the weights are still trained normally.
- *Why a smaller learning rate for the pretrained weights?* The new head starts from random weights and must learn fast (1e-3). The pretrained weights are already good, so we only adjust them (1e-4) instead of destroying them.
- *Why does fine-tuning overfit after 3 epochs?* 11 million weights adapted on 1 050 paintings: the training accuracy reaches 0.98 within 7 epochs. Early stopping keeps the weights of epoch 3.
- *Why is a logistic regression on the features almost as good?* The ImageNet features already separate the styles. A linear model on top is enough, and it is the same model as the "head only" variant, trained differently.
- *Why 128 pixels and not 224?* Our subset is stored at 128 pixels. 224 is the ImageNet resolution and might add a few points, at about 3 times the computation per painting on a kiosk. We did not measure it in the notebooks: it is a lever to try.

**Production**

- *Would it work on photos taken in the gallery?* Not tested. WikiArt images are clean scans; a visitor's photo has reflections, angles and different lighting. We would need a test set of real photos, and augmentation such as crops, colour changes and perspective.
- *What if the museum adds a style?* With frozen features and a logistic regression, label a few hundred paintings and retrain in seconds on a CPU. Fine-tuning would need a new GPU run.
- *What does the guide do when it is unsure?* The ResNet is less confident when it is wrong (0.70 against 0.87). With a threshold of 0.7 it answers 77 % of the paintings and is right on 86 % of them. In production, the threshold must be set on the validation set.
- *What about the natural-language search of the brief?* It was outside the three deliverables (stretch issue #13). A text–image model such as CLIP would be the natural next step.
- *Is it reproducible?*
  - Seeds are fixed, and every notebook gives the same numbers when re-run.
  - The README gives the order and the commands.
  - A teammate reproduces the results from a fresh clone (#12).
