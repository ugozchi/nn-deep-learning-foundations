# Oral preparation

Slides: [`museum_guide.pptx`](museum_guide.pptx), 11 slides for 10 minutes, with speaker notes on every slide. Every number comes from `results/`.

## Speakers and timing (proposal)

| Part | Slides | Topic | Speaker | Time |
| --- | --- | --- | --- | --- |
| 1 | 1–4 | The question, the data, the classical baselines | Féliz | 3 min |
| 2 | 5–7 | The MLP, monitoring, diagnosis | Ugo | 3 min 30 |
| 3 | 8–11 | The ResNet, the errors, the cost, the recommendation | Thomas | 3 min 30 |

Slides 10 and 11 show the CPU benchmark (#9) and the recommendation (#10). Two timed rehearsals before the 30th.

## Likely questions

**Data and protocol**

- *Why these five styles?* They differ visibly in palette, brushwork and composition, so 300 paintings per style are enough to learn, and they leave room for a gap between a classical model and a CNN. Style is also what the guide needs to tell the story of a movement.
- *How did you avoid overfitting the test set?*
  - Every choice (image size, `C`, architecture, regularisation, fine-tuning variant) was made on the validation set.
  - The test set was used once per final model.
  - The normalisation statistics come from the training images only.
- *Is 0.804 really better than 0.769?* Probably a little, but we cannot measure it precisely. On 225 test paintings an accuracy has a standard error of about 3 points, and the gap is 3.5 points. The validation set points the same way (0.790 ± 0.017 over 3 seeds, against 0.733). 0.44 → 0.80, on the other hand, is far beyond the noise.
- *Why 3 seeds?* One run on 225 validation paintings can move by a few points with the seed alone (initial weights, batch order). The mean and the standard deviation tell whether a difference is real.
- *Why macro-F1?* It is the mean of the F1 scores of the five styles, so every style counts equally. Our classes are balanced, so it stays close to the accuracy, but it would reveal a style that is ignored if the collection became unbalanced.

**The MLP**

- *Why is the MLP so weak?* It has one weight per pixel. The same brushwork or shape at another place in the painting uses other weights, and the network has no notion of local texture. Regularisation shrinks the train–validation gap (0.62 → 0.07) but not the ceiling (about 0.40). Its per-class profile is the same as that of the colour models.
- *Why 32 × 32 pixels?* On raw pixels, 16, 32 and 64 pixels gave the same accuracy (baseline notebook). 32 keeps some detail with 3 072 inputs, so the first layer has 1.6 million weights instead of 6.3 million at 64 pixels.
- *How do you know it is overfitting?*
  - The training accuracy reaches 0.99 while the validation accuracy stays at 0.36.
  - The validation loss is lowest at epoch 4, then rises to 4.0 while the training loss keeps falling.
- *Why does the sigmoid learn more slowly?* Its derivative is at most 0.25, and 30 % of the first-layer outputs are saturated, where the derivative is almost 0. Backpropagation multiplies these derivatives, so the first layer gets a gradient 68 times smaller than the output layer. The result: best epoch 18 instead of 4.
- *Why Adam with lr 1e-4?* At 1e-2 the loss explodes at the first epoch (10.3), then oscillates. At 1e-6 it barely moves in 30 epochs. 1e-4 trains steadily (monitoring notebook).

**The ResNet**

- *Why keep the batch norm in `eval()` mode?* With 1 050 paintings and batches of 64, the batch statistics are noisy. As in TP07, we keep the statistics learned on ImageNet. ResNet-18 has no dropout, so `eval()` changes nothing else, and the weights are still trained normally.
- *Why a smaller learning rate for the pretrained weights?* The new head starts from random weights and must learn fast (1e-3). The pretrained weights are already good, so we only adjust them (1e-4) instead of destroying them.
- *Why does fine-tuning overfit after 3 epochs?* 11 million weights adapted on 1 050 paintings: the training accuracy reaches 0.98 within 7 epochs. Early stopping keeps the weights of epoch 3.
- *Why is a logistic regression on the features almost as good?* The ImageNet features already separate the styles. A linear model on top is enough, and it is the same model as the "head only" variant, trained differently.
- *Why 128 pixels and not 224?* Our subset is stored at 128 pixels. 224 is the ImageNet resolution and might add a few points. It has 3 times more pixels, and the benchmark measures 2.1 times the CPU time per painting (9.7 ms against 4.6 ms on one thread). We did not train at 224: it is a lever to try.

**The recommendation**

- *Why not the fine-tuned ResNet, which is more accurate?* It is 3.5 points better on the test set, about one standard error, and it costs the same on the kiosk. But every change of the collection would need a GPU run that overfits after 3 epochs and must be watched, and a new 45 MB model on every kiosk. With frozen features, only a logistic regression of 2 565 weights is refitted, in seconds on a CPU. If the museum prefers a few points to simple updates, fine-tuning is the alternative, at the same kiosk cost.
- *Is 4.8 ms realistic on a modest kiosk?* It is one core of a recent laptop, so a kiosk will be slower. Even on a CPU 20 times slower it stays under 0.1 s per painting, which a visitor does not notice. Before deploying, we would run the benchmark notebook once on the kiosk.
- *So a classical model does not do the job?* Not on pixels or colours: 0.37 to 0.40, wrong 6 times out of 10. But our recommendation is half classical: the network gives the features, and the model we retrain is a logistic regression.
- *Why not a smaller network, such as MobileNet?* Not tested. The ResNet-18 already runs in under 5 ms on one core, so a smaller network would save time we do not need. It would matter on much weaker hardware.

**Production**

- *Would it work on photos taken in the gallery?* Not tested. WikiArt images are clean scans; a visitor's photo has reflections, angles and different lighting. We would need a test set of real photos, and augmentation such as crops, colour changes and perspective.
- *What if the museum adds a style?* With frozen features and a logistic regression, label a few hundred paintings and retrain in seconds on a CPU. Fine-tuning would need a new GPU run.
- *What does the guide do when it is unsure?* The ResNet is less confident when it is wrong (0.72 against 0.87). With a threshold of 0.7 it answers 78 % of the paintings and is right on 87 % of them. In production, the threshold must be set on the validation set.
- *What about the natural-language search of the brief?* It was outside the three deliverables (stretch issue #13). A text–image model such as CLIP would be the natural next step.
- *Is it reproducible?*
  - Seeds are fixed, and every notebook gives the same numbers when re-run on the same machine.
  - On another machine the two networks move by about one point (MLP 0.431 instead of 0.440, ResNet 0.796 instead of 0.804, from a fresh clone on another laptop), within the ± 3 points of the test set. The logistic regressions give exactly the same numbers.
  - The README gives the order and the commands.
  - A teammate reproduces the results from a fresh clone (#12).
