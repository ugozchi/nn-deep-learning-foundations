# QCM d'entraînement · Neural Networks & Deep Learning Foundations

58 questions, dans l'ordre des séances. **Une seule bonne réponse** par question.
Le corrigé (avec le "pourquoi") est **à la fin** : fais d'abord tout le QCM sans regarder.

> Pour faire réviser un groupe : projette une séance à la fois, 1 minute par question, puis corrige ensemble avec la colonne "Pourquoi".
> Les explications détaillées sont dans [`01_master_revision.md`](01_master_revision.md).

---

## Séance 1 · Le perceptron

**1.** Perceptron avec $w = (1, 1)$, $b = -1.5$. Pour $x = (1, 1)$, $\hat y$ vaut :
- A) 0
- B) 1
- C) 0.5
- D) −0.5

**2.** Perceptron avec $w = (2, 1)$, $b = -2$. Où la frontière coupe-t-elle l'axe $x_1$ ?
- A) $(2, 0)$
- B) $(1, 0)$
- C) $(0, 1)$
- D) $(-1, 0)$

**3.** Par rapport à la frontière de décision, le vecteur $w$ est :
- A) parallèle à la frontière
- B) orthogonal, et pointe vers la classe 1
- C) orthogonal, et pointe vers la classe 0
- D) sans lien avec la frontière

**4.** On ne modifie que le biais $b$. La frontière :
- A) tourne autour de l'origine
- B) se déplace parallèlement à elle-même
- C) ne bouge pas
- D) change de pente

**5.** Faux positif ($y = 0$, $\hat y = 1$), $\eta = 1$. La règle du perceptron fait :
- A) $w \leftarrow w + x$, $b \leftarrow b + 1$
- B) $w \leftarrow w - x$, $b \leftarrow b - 1$
- C) rien
- D) $w \leftarrow -w$

**6.** Le théorème de Novikoff garantit :
- A) la convergence sur n'importe quelles données
- B) un nombre fini d'erreurs si les données sont linéairement séparables
- C) la droite de marge maximale
- D) des probabilités bien calibrées

**7.** Pourquoi ne peut-on pas entraîner un perceptron par descente de gradient ?
- A) il a trop de paramètres
- B) la fonction marche a une dérivée nulle partout (non définie en 0)
- C) la loss n'existe pas pour 2 classes
- D) les poids doivent rester positifs

## Séance 2 · Le neurone sigmoïde

**8.** Sachant $\sigma(2) = 0.881$, combien vaut $\sigma(-2)$ ?
- A) 0.881
- B) 0.119
- C) −0.881
- D) 0.5

**9.** La valeur maximale de $\sigma'(z)$ est :
- A) 1
- B) 0.5
- C) 0.25
- D) 0.1

**10.** Vraie classe $y = 1$, prédiction $a = 0.1$. La log loss (log naturel) vaut :
- A) 0.105
- B) 0.693
- C) 2.303
- D) 0.9

**11.** Gradient de la log loss d'un neurone sigmoïde par rapport à $w_j$ :
- A) $(a - y)\,x_j$
- B) $(a - y)\,a(1-a)\,x_j$
- C) $(y - a)$
- D) $a(1-a)\,x_j$

**12.** Pourquoi PyTorch recommande `BCEWithLogitsLoss` plutôt que `sigmoid` puis log loss ?
- A) c'est plus court à écrire
- B) stabilité numérique : évite $\log(0)$ quand $|z|$ est grand
- C) elle ajoute une régularisation
- D) elle accepte plus de 2 classes

**13.** La frontière de décision d'une régression logistique est :
- A) une courbe quelconque
- B) une droite (un hyperplan)
- C) un cercle
- D) cela dépend des données

**14.** Pourquoi ne pas utiliser l'erreur quadratique (MSE) avec une sortie sigmoïde ?
- A) non convexe, et gradient très faible quand le modèle est très faux
- B) trop lente à calculer
- C) elle n'est pas dérivable
- D) elle ne marche que pour la régression linéaire

## Séance 3 · Premier réseau et backpropagation

**15.** Nombre de paramètres d'un réseau dense 784-128-10 :
- A) 101 632
- B) 101 770
- C) 100 480
- D) 1 290

**16.** `nn.Linear(3, 5).weight.shape` vaut :
- A) (3, 5)
- B) (5, 3)
- C) (5,)
- D) (3,)

**17.** Deux `nn.Linear` à la suite **sans** activation entre elles équivalent à :
- A) un réseau non linéaire à 2 couches
- B) une seule couche linéaire
- C) une sigmoïde
- D) un perceptron multicouche

**18.** En backpropagation (réseau 2 couches, sigmoïdes), $dZ^{[1]}$ vaut :
- A) $A^{[1]} - Y$
- B) $W^{[2]\top}dZ^{[2]} \odot \sigma'(Z^{[1]})$
- C) $dZ^{[2]}A^{[1]\top}$
- D) $W^{[1]}dZ^{[2]}$

**19.** Le réseau XOR du cours calcule :
- A) AND(OR, NAND)
- B) OR(AND, NAND)
- C) NAND(OR, AND)
- D) AND(OR, OR)

**20.** Réseau 2-2-1 : $a^{[1]} = (0.5, 0.5)$, $a^{[2]} = 0.5$, $y = 1$, log loss. $dW^{[2]}$ vaut :
- A) $(-0.5, -0.5)$
- B) $(-0.25, -0.25)$
- C) $(0.25, 0.25)$
- D) $(-0.125, -0.125)$

## Séance 4 · PyTorch en profondeur

**21.** Comparée à $\text{softmax}(2, 1, 0)$, la sortie de $\text{softmax}(12, 11, 10)$ est :
- A) identique
- B) plus confiante
- C) moins confiante
- D) de somme 10

**22.** Un réseau non entraîné sur 10 classes a une loss de départ d'environ :
- A) 0
- B) 0.693
- C) 2.303
- D) 10

**23.** `nn.CrossEntropyLoss` attend :
- A) des probabilités et des labels one-hot
- B) des logits et des labels int64 de shape `(B,)`
- C) des logits et des labels float
- D) des probabilités et des labels int64

**24.** Pourquoi appeler `optimizer.zero_grad()` à chaque pas ?
- A) pour remettre les poids à zéro
- B) parce que `backward()` additionne les gradients dans `.grad`
- C) pour libérer la mémoire GPU
- D) pour réinitialiser le learning rate

**25.** 1000 exemples d'entraînement, `batch_size=64`. Nombre de pas par epoch :
- A) 15
- B) 16
- C) 64
- D) 1000

**26.** Quelle addition provoque une erreur de broadcasting ?
- A) `(2,3) + (3,)`
- B) `(2,3) + (2,1)`
- C) `(2,3) + (2,)`
- D) `(2,3) + (1,3)`

**27.** Sur quel ensemble choisit-on les hyperparamètres (lr, nombre d'epochs, architecture) ?
- A) train
- B) validation
- C) test
- D) les trois ensemble

**28.** `x = torch.tensor(3.0, requires_grad=True); f = x**2 + 3*x; f.backward()`. `x.grad` vaut :
- A) 18
- B) 9
- C) 6
- D) 3

**29.** Le gradient de softmax + cross-entropy par rapport aux logits vaut :
- A) $p - y$
- B) $y - p$
- C) $-\log p$
- D) $p(1-p)$

**30.** Que manque-t-il dans cette évaluation ?
```python
correct = 0
for xb, yb in val_loader:
    correct += (model(xb).argmax(dim=1) == yb).sum().item()
```
- A) `optimizer.step()`
- B) `model.eval()` et `torch.no_grad()`
- C) un `softmax` avant `argmax`
- D) `loss.backward()`

## Séance 5 · Vision, activations, initialisation

**31.** Un réseau à 6 couches cachées sigmoïdes sur FashionMNIST (SGD, lr 0.2), après 5 epochs :
- A) plus de 86 %
- B) environ 86 %
- C) un peu moins de 86 %
- D) environ 10 %

**32.** La variance de l'initialisation He (Kaiming) vaut :
- A) $1/n_\text{in}$
- B) $2/n_\text{in}$
- C) $2/(n_\text{in} + n_\text{out})$
- D) 1

**33.** Tous les poids d'une couche sont initialisés à la même valeur. Problème :
- A) vanishing gradient
- B) symétrie : tous les neurones restent identiques pour toujours
- C) explosion des activations
- D) aucun problème

**34.** Un neurone ReLU est « mort » quand :
- A) $z > 0$ pour toutes les entrées
- B) $z < 0$ pour toutes les entrées : sortie 0 et gradient nul pour toujours
- C) ses poids sont très petits
- D) sa sortie vaut toujours 1

**35.** Quelle activation utilise-t-on dans les couches cachées des transformers ?
- A) sigmoïde
- B) GELU
- C) softmax
- D) marche

**36.** Chaîne de 5 neurones sigmoïdes, tous les poids à 1. Majorant de $|\partial a^{[5]}/\partial a^{[0]}|$ :
- A) 0.25
- B) $1/1024$
- C) $1/20$
- D) 1

**37.** Poids $\mathcal{N}(0, 1)$, $n_\text{in} = 784$, entrées de variance 1. L'écart-type de $z$ est :
- A) 1
- B) 784
- C) 28
- D) 0.036

## Séance 6 · Mieux entraîner

**38.** $w = 2$, gradient des données $0.5$, $\eta = 0.1$, weight decay $\lambda = 0.5$. Nouveau $w$ :
- A) 1.95
- B) 1.85
- C) 1.90
- D) 2.05

**39.** `nn.Dropout(p=0.5)` en mode entraînement. Une activation qui vaut 4 et qui est **conservée** devient :
- A) 2
- B) 4
- C) 8
- D) 0

**40.** Dans `nn.Dropout(p=0.2)`, `p` est :
- A) la probabilité de garder un neurone
- B) la probabilité de mettre un neurone à zéro
- C) le facteur de mise à l'échelle
- D) le learning rate du dropout

**41.** Train et validation sont plates à 2.30 depuis le début (10 classes). Diagnostic :
- A) overfitting
- B) le réseau prédit ~1/10 partout : il n'apprend rien
- C) learning rate trop grand
- D) underfitting léger

**42.** La train loss descend, la validation loss remonte. Diagnostic :
- A) underfitting
- B) overfitting
- C) learning rate trop petit
- D) bug de chargement des données

**43.** La loss devient `nan` après quelques epochs. Premier changement :
- A) augmenter le learning rate
- B) diviser le learning rate par 10
- C) ajouter des couches
- D) supprimer le set de validation

**44.** Taille du **premier** pas d'Adam ($\varepsilon$ négligé) :
- A) $\eta|g|$
- B) $\eta$
- C) $\eta g^2$
- D) 0

**45.** Batch norm sur le batch $z = (1, 3, 3, 9)$ :
- A) $\mu_B = 4$, $\sigma_B^2 = 9$
- B) $\mu_B = 4$, $\sigma_B^2 = 12$
- C) $\mu_B = 3$, $\sigma_B^2 = 9$
- D) $\mu_B = 4$, $\sigma_B^2 = 3$

**46.** Gradient clipping avec $g = (3, 4)$ et $c = 1$ :
- A) $(3, 4)$
- B) $(0.6, 0.8)$
- C) $(0.3, 0.4)$
- D) $(1, 1)$

**47.** En mode `eval()`, une couche de batch norm utilise :
- A) la moyenne et la variance du batch courant
- B) les statistiques courantes (running mean et running var)
- C) $\mu = 0$ et $\sigma = 1$
- D) aucune normalisation

**48.** Momentum avec $\beta = 0.9$ et un gradient constant : les pas deviennent jusqu'à :
- A) 0.9 fois plus grands
- B) 2 fois plus grands
- C) 10 fois plus grands
- D) 100 fois plus grands

## Séance 7 · Les CNN

**49.** Image $28 \times 28$, noyau $3 \times 3$, padding 1, stride 2. Taille de sortie :
- A) 13
- B) 14
- C) 26
- D) 28

**50.** Nombre de paramètres de `nn.Conv2d(3, 16, 3)` :
- A) 432
- B) 448
- C) 144
- D) 160

**51.** Nombre de paramètres de `nn.MaxPool2d(2)` :
- A) 4
- B) 1
- C) 0
- D) cela dépend du nombre de canaux

**52.** Champ récepteur de trois convolutions $3 \times 3$ empilées (stride 1) :
- A) 3
- B) 5
- C) 7
- D) 9

**53.** Un CNN « plain » de 56 couches a une erreur **d'entraînement** plus haute qu'un réseau de 20 couches. C'est :
- A) de l'overfitting
- B) un problème d'optimisation (la dégradation)
- C) un bug du code
- D) un effet du dropout

**54.** Transfer learning avec peu d'images, proches de photos ImageNet. On commence par :
- A) tout réentraîner avec un grand learning rate
- B) geler le backbone et entraîner une nouvelle tête
- C) repartir de poids aléatoires
- D) supprimer les premières couches

## Séance 8 · RNN, LSTM, attention

**55.** Nombre de paramètres de `nn.RNN(input_size=10, hidden_size=20)` :
- A) 600
- B) 620
- C) 640
- D) 2560

**56.** On multiplie tous les scores d'attention par 10. Les poids $\alpha$ deviennent :
- A) plus uniformes
- B) plus concentrés (presque one-hot)
- C) inchangés
- D) de somme 10

**57.** `nn.LSTM` renvoie :
- A) `out`
- B) `out, h_n`
- C) `out, (h_n, c_n)`
- D) `(h_n, c_n)`

**58.** Pourquoi un LSTM garde-t-il l'information plus longtemps qu'un RNN simple ?
- A) parce qu'il a plus de paramètres
- B) le cell state est mis à jour de façon additive : $\partial c_t/\partial c_{t-1} = \text{diag}(f_t)$
- C) parce qu'il utilise tanh au lieu de sigmoïde
- D) parce qu'il n'utilise pas la backpropagation

---
---

## Corrigé

| N° | Rép. | Pourquoi |
| --- | --- | --- |
| 1 | **B** | $z = 1 + 1 - 1.5 = 0.5 \ge 0$ → 1 (c'est AND) |
| 2 | **B** | $x_1 = -b/w_1 = 2/2 = 1$ (l'autre intersection est $(0, 2)$) |
| 3 | **B** | $w^\top(x_A - x_B) = 0$ pour deux points de la droite ; $w$ pointe vers $z > 0$ |
| 4 | **B** | $b$ décale la droite sans la tourner ; l'orientation vient de $w$ |
| 5 | **B** | erreur $y - \hat y = -1$ → $w$ s'éloigne de $x$ |
| 6 | **B** | séparable → nombre fini d'erreurs (≤ $(R/\gamma)^2$) ; rien sur le non séparable |
| 7 | **B** | dérivée nulle → aucune information sur la direction à prendre |
| 8 | **B** | symétrie $\sigma(-z) = 1 - \sigma(z) = 0.119$ |
| 9 | **C** | $\sigma' = \sigma(1-\sigma)$, max en $z = 0$ : $0.5 \times 0.5$ |
| 10 | **C** | $-\ln 0.1 = \ln 10 = 2.303$ |
| 11 | **A** | le $a(1-a)$ de la sigmoïde s'annule avec celui de la log loss : erreur × entrée |
| 12 | **B** | en float32, $\sigma(-200) = 0$ et $\log 0 = -\infty$ ; la forme stable donne 200 |
| 13 | **B** | $a \ge 0.5 \iff w^\top x + b \ge 0$ |
| 14 | **A** | facteur $a(1-a) \approx 0$ quand très faux, et coût non convexe |
| 15 | **B** | $784 \times 128 + 128 + 128 \times 10 + 10 = 100\,480 + 1\,290$ |
| 16 | **B** | `weight` a la shape `(n_out, n_in)` |
| 17 | **B** | $W_2(W_1x + b_1) + b_2 = W'x + b'$ |
| 18 | **B** | on renvoie l'erreur avec $W^{[2]\top}$ puis on multiplie par $\sigma'$ |
| 19 | **A** | XOR = "au moins un" ET "pas les deux" |
| 20 | **B** | $dz^{[2]} = 0.5 - 1 = -0.5$ ; $dW^{[2]} = -0.5 \times (0.5, 0.5)$ |
| 21 | **A** | ajouter une constante à tous les logits ne change pas la softmax |
| 22 | **C** | $p = 1/10$ partout → $-\ln(1/10) = \ln 10$ |
| 23 | **B** | elle applique log-softmax elle-même ; labels = indices de classe |
| 24 | **B** | sinon le gradient du pas précédent s'ajoute au nouveau |
| 25 | **B** | $\lceil 1000/64 \rceil = \lceil 15.6 \rceil = 16$ (le dernier batch a 40 exemples) |
| 26 | **C** | depuis la droite : 3 contre 2, ni égaux ni 1 |
| 27 | **B** | le test sert une seule fois, à la fin |
| 28 | **B** | $f' = 2x + 3 = 9$ |
| 29 | **A** | prédiction moins vérité, comme $(a - y)$ en binaire |
| 30 | **B** | mode évaluation (dropout, batch norm) et pas de graphe ; `argmax` sur les logits suffit |
| 31 | **D** | vanishing gradient : le niveau du hasard, loss bloquée à 2.30 |
| 32 | **B** | le 2 compense la ReLU qui met la moitié des valeurs à 0 |
| 33 | **B** | même forward, même gradient, même mise à jour : symétrie jamais cassée |
| 34 | **B** | ReLU' = 0 si $z < 0$ : plus aucun gradient n'atteint $w$ et $b$ |
| 35 | **B** | GELU = ReLU lisse (BERT, GPT) ; tanh dans les RNN |
| 36 | **B** | $(1/4)^5 = 1/1024$ |
| 37 | **C** | $\text{Var}(z) = 784 \times 1 \times 1$ → std $= 28$ → sigmoïde saturée |
| 38 | **B** | $(1 - 0.05) \times 2 - 0.1 \times 0.5 = 1.9 - 0.05$ |
| 39 | **C** | inverted dropout : $4 / (1 - 0.5) = 8$ |
| 40 | **B** | `p` = probabilité d'éteindre ; les valeurs gardées sont ×$1/(1-p)$ |
| 41 | **B** | $\ln 10 = 2.30$ = prédiction uniforme ; chercher un bug, sur-apprendre un batch |
| 42 | **B** | l'écart train/val grandit : early stopping, régularisation |
| 43 | **B** | nan = lr trop grand / explosion ; aussi gradient clipping |
| 44 | **B** | $\hat m = g$, $\hat s = g^2$ → pas $= \eta\,g/\lvert g \rvert$ |
| 45 | **A** | $\mu = 16/4$ ; $\sigma^2 = (9 + 1 + 1 + 25)/4 = 36/4$ |
| 46 | **B** | $\|g\| = 5 > 1$ → $g \times 1/5$ : direction gardée |
| 47 | **B** | on peut prédire une seule image : pas de batch pour calculer des stats |
| 48 | **C** | $v \to g/(1 - \beta) = 10g$ |
| 49 | **B** | $\lfloor(28 + 2 - 3)/2\rfloor + 1 = 13 + 1$ |
| 50 | **B** | $16 \times (3 \times 3 \times 3 + 1) = 16 \times 28$ |
| 51 | **C** | le pooling ne fait que prendre un max (ou une moyenne) |
| 52 | **C** | $r = 1 \to 3 \to 5 \to 7$ |
| 53 | **B** | c'est l'erreur de **train** qui est mauvaise : d'où ResNet |
| 54 | **B** | puis éventuellement dégeler le dernier bloc avec un petit lr |
| 55 | **C** | $20 \times 10 + 20 \times 20 + 2 \times 20 = 200 + 400 + 40$ (LSTM : ×4 = 2 560) |
| 56 | **B** | la softmax amplifie les écarts : $(2,1,3) \times 10$ → $\alpha_3 \approx 0.99995$ |
| 57 | **C** | le LSTM renvoie aussi son cell state |
| 58 | **B** | chemin additif sans matrice de poids ni tanh' : comme le $+x$ de ResNet |

**Barème indicatif** : 50+ = prêt ; 40–49 = relire les pièges QCM des séances où tu as raté ; < 40 = refaire les exercices du master doc à la main.
