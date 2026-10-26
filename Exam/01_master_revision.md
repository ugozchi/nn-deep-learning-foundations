# Master de révision · Neural Networks & Deep Learning Foundations

Toutes les séances, tous les exercices des slides, corrigés pas à pas et expliqués simplement.

> **Comment lire ce fichier** : ouvre-le en aperçu Markdown (VS Code : `Cmd + Shift + V`) pour voir les formules.
> Les nombres sont écrits avec un point (`0.5`), comme dans le cours et l'examen.

---

## Sommaire

0. [L'examen et comment utiliser ce document](#0-lexamen-et-comment-utiliser-ce-document)
1. [Séance 1 · Le perceptron](#séance-1--le-perceptron)
2. [Séance 2 · Le neurone sigmoïde (régression logistique)](#séance-2--le-neurone-sigmoïde-régression-logistique)
3. [Séance 3 · Premier réseau et backpropagation](#séance-3--premier-réseau-et-backpropagation)
4. [Séance 4 · PyTorch en profondeur](#séance-4--pytorch-en-profondeur)
5. [Séance 5 · Vision, activations, initialisation](#séance-5--vision-activations-initialisation)
6. [Séance 6 · Mieux entraîner un réseau](#séance-6--mieux-entraîner-un-réseau)
7. [Séance 7 · Les réseaux convolutifs (CNN)](#séance-7--les-réseaux-convolutifs-cnn)
8. [Séance 8 · RNN, LSTM, attention](#séance-8--rnn-lstm-attention)
9. [Formulaire express](#9-formulaire-express)
10. [Le code PyTorch à savoir lire](#10-le-code-pytorch-à-savoir-lire)
11. [Problème → solution : la grande table](#11-problème--solution--la-grande-table)

Le fichier [`02_qcm_entrainement.md`](02_qcm_entrainement.md) contient 58 questions type QCM, avec les réponses à la fin.

---

## 0. L'examen et comment utiliser ce document

**Ce que dit la dernière slide du cours (séance 8) :**

- 90 minutes, **sans IA**.
- **Lire et écrire du code** PyTorch, comme dans les TP.
- **Connaissances** : les formules et les idées des 8 séances.
- **Questions de réflexion** : expliquer, diagnostiquer, choisir, justifier.
- Conseil du prof : **refaire à la main les exercices de chaque séance** (un forward, un gradient, une mise à jour, une convolution, une taille de sortie, un champ récepteur, des poids d'attention).

**Pour faire cours aux camarades**, chaque séance suit le même plan :

| Bloc | À quoi il sert |
| --- | --- |
| 🧒 **L'histoire** | expliquer l'idée en 1 minute, sans maths |
| 📌 **Ce qu'il faut savoir** | les formules et les phrases à connaître par cœur |
| ✏️ **Exercices** | énoncé, puis **recette** (la méthode), puis **correction** pas à pas |
| ⚠️ **Pièges QCM** | les erreurs classiques que l'examen adore |

> Astuce pédagogique : donne l'énoncé, laisse 3 minutes, puis déroule la recette au tableau avant de donner la correction.

**Le fil rouge du cours en une phrase** : *un modèle, une loss, son gradient ; et quand le modèle bloque, une meilleure architecture. Chaque correctif (ReLU, He, batch norm, ResNet, LSTM) sert à laisser passer le gradient.*

| Séance | Ce qu'on construit | Sa limite (corrigée à la séance suivante) |
| --- | --- | --- |
| 1 | Perceptron, règle d'apprentissage | XOR impossible, sortie 0/1, pas de gradient |
| 2 | Neurone sigmoïde, log loss, gradient $(a-y)x$ | toujours une seule droite |
| 3 | Réseau à couches, backpropagation, autograd | besoin d'un pipeline propre |
| 4 | PyTorch : tenseurs, modules, loss, optimiseur, boucle | les réseaux sigmoïdes profonds n'apprennent pas |
| 5 | ReLU, vanishing gradient, init Xavier / He | les gros réseaux sur-apprennent |
| 6 | Weight decay, dropout, early stopping, batch norm, Adam | les couches denses ignorent la structure des images |
| 7 | Convolution, pooling, champ récepteur, ResNet, transfer learning | taille fixe, pas de notion d'ordre |
| 8 | RNN, LSTM, attention, embeddings | mémoire courte et calcul séquentiel → transformers |

---

## Séance 1 · Le perceptron

### 🧒 L'histoire

Imagine un **videur de boîte de nuit**. Il regarde quelques infos sur toi (les **entrées** $x_1, x_2$). Chaque info a une importance (un **poids** $w_1, w_2$). Il additionne les points, ajoute son humeur du jour (le **biais** $b$). Si le total est $\ge 0$ tu rentres (1), sinon non (0). C'est **tout** le perceptron.

Et pour **apprendre** : à chaque fois qu'il se trompe, il corrige un peu ses poids dans la bonne direction.

### 📌 Ce qu'il faut savoir

**Le modèle**

$$z = w_1x_1 + w_2x_2 + b = w^\top x + b \qquad \hat y = \begin{cases}1 & \text{si } z \ge 0\\ 0 & \text{si } z < 0\end{cases}$$

- $z$ = le **score**, $\hat y$ = la **prédiction**, $y$ = la **vraie étiquette**.
- Attention : $z = 0$ donne $\hat y = 1$ (c'est un $\ge$).
- Historique : McCulloch et Pitts (1943) = somme pondérée + seuil ; Rosenblatt (1958) = le perceptron qui **apprend** ses poids.

**La géométrie**

- La **frontière de décision** est l'ensemble des points où $z = 0$ : $w_1x_1 + w_2x_2 + b = 0$. En 2D c'est une **droite**, en $n$ dimensions un **hyperplan**.
- $w$ est **orthogonal** (perpendiculaire) à la droite et **pointe vers la classe 1**.
  - Preuve : deux points $x_A, x_B$ sur la droite donnent $w^\top x_A + b = 0$ et $w^\top x_B + b = 0$, donc $w^\top(x_A - x_B) = 0$.
- $b$ **décale** la droite **parallèlement** (sans la tourner). Si $b = 0$, la droite passe par l'origine.
- Intersections avec les axes : $x_1 = -b/w_1$ (quand $x_2 = 0$) et $x_2 = -b/w_2$ (quand $x_1 = 0$).

**La règle d'apprentissage** (on part de $w = 0$, $b = 0$)

$$w \leftarrow w + \eta\,(y - \hat y)\,x \qquad b \leftarrow b + \eta\,(y - \hat y)$$

| Cas | $y - \hat y$ | Effet sur $w$ | En image |
| --- | --- | --- | --- |
| correct | 0 | rien | |
| **faux négatif** ($y=1$, $\hat y=0$) | +1 | $w + \eta x$ | $w$ se tourne **vers** $x$ |
| **faux positif** ($y=0$, $\hat y=1$) | −1 | $w - \eta x$ | $w$ s'**éloigne** de $x$ |

- Pourquoi ça marche : après une erreur $e = y - \hat y$, le nouveau score du même point est $z' = z + \eta e(\|x\|^2 + 1)$. Comme $\|x\|^2 + 1 > 0$, le score monte si $e = +1$ et descend si $e = -1$ : **toujours dans le bon sens** (il faut parfois plusieurs corrections).
- **Epoch** = un passage complet sur tout le dataset. On s'arrête quand une epoch entière n'a **aucune erreur**.
- **Théorème de Novikoff (1962)** : si les données sont **linéairement séparables**, le perceptron fait un **nombre fini d'erreurs** puis s'arrête (au plus $(R/\gamma)^2$ erreurs, avec $\|x\| \le R$ et $\gamma$ la marge). Il ne dit **rien** sur les données non séparables.

**Vectorisation** : avec les $m$ exemples en lignes de $X$ (shape $(m, n)$), `X @ w + b` calcule les $m$ scores d'un coup (shape $(m,)$), sans boucle Python.

**Les 4 limites du perceptron** (à connaître par cœur)

| Limite | Correctif | Quand |
| --- | --- | --- |
| 1. **XOR** : une seule droite ne suffit pas | plusieurs neurones en couches + backprop | Séance 3 |
| 2. **Données non séparables** : la règle ne s'arrête jamais, pas de "meilleure" droite | minimiser une loss (la log loss) | Séance 2 |
| 3. **Sortie 0 ou 1** : aucune confiance ($z = 0.01$ et $z = 100$ donnent tous les deux 1) | la sigmoïde → une probabilité | Séance 2 |
| 4. **Pas de gradient** : la marche a une dérivée nulle partout | la sigmoïde est dérivable → descente de gradient | Séance 2 |

### ✏️ Exercice 1 · Un perceptron qui calcule AND

> Poids $w = (1, 1)$, biais $b = -1.5$.
> 1. Remplis $z$ et $\hat y$ pour les 4 entrées. Est-ce AND ?
> 2. Trouve des poids et un biais pour OR. (Indice : garde $w = (1,1)$, change seulement $b$.)
> 3. Bonus : NAND (NOT AND).

**Recette** : pour chaque ligne, calcule $z = w_1x_1 + w_2x_2 + b$, puis applique la marche ($\hat y = 1$ si $z \ge 0$).

**Correction**

1. $z = x_1 + x_2 - 1.5$

| $x_1$ | $x_2$ | $z$ | $\hat y$ | AND |
| --- | --- | --- | --- | --- |
| 0 | 0 | −1.5 | 0 | 0 |
| 0 | 1 | −0.5 | 0 | 0 |
| 1 | 0 | −0.5 | 0 | 0 |
| 1 | 1 | 0.5 | 1 | 1 |

Oui, c'est AND : il faut **les deux** à 1 pour dépasser 1.5.

2. **OR** : on veut que **un seul** 1 suffise.
   - $(0,0)$ doit donner 0 : $z = b < 0$.
   - $(0,1)$ doit donner 1 : $z = 1 + b \ge 0$, donc $b \ge -1$.
   - N'importe quel $b \in [-1, 0[$ marche. On prend **$b = -0.5$** : $z = -0.5,\ 0.5,\ 0.5,\ 1.5$ → $\hat y = 0, 1, 1, 1$. ✅

3. **NAND** = l'inverse de AND : on **retourne tous les signes** : $w = (-1, -1)$, $b = 1.5$.
   $z = 1.5,\ 0.5,\ 0.5,\ -0.5$ → $\hat y = 1, 1, 1, 0$. ✅

### ✏️ Exercice 2 · Tracer la droite de l'exercice 1

> 1. Équation de la droite AND et ses intersections avec les axes.
> 2. Trace-la avec les 4 points, trace $w$. De quel côté est la classe 1 ?
> 3. Vérifie que $w$ est orthogonal à la droite avec le vecteur directeur $(1, -1)$.
> 4. Trace la droite OR. Qu'est-ce qui a changé, qu'est-ce qui n'a pas changé ?

**Correction**

1. $x_1 + x_2 - 1.5 = 0$, soit $x_2 = 1.5 - x_1$. Intersections : $(1.5, 0)$ et $(0, 1.5)$ (calcul : $-b/w_1 = 1.5$).
2. La droite coupe le carré en diagonale ; seul le point $(1,1)$ est au-dessus. $w = (1,1)$ part de la droite vers le haut-droite, donc **vers $(1,1)$** : c'est le côté classe 1.
3. Produit scalaire $w \cdot (1,-1) = 1 \times 1 + 1 \times (-1) = 0$ → orthogonal. ✅
4. Droite OR : $x_1 + x_2 - 0.5 = 0$, intersections $(0.5, 0)$ et $(0, 0.5)$.
   - **Changé** : la **position** (on a changé $b$) ; la droite s'est rapprochée de l'origine.
   - **Pas changé** : l'**orientation** (même $w$) → droite **parallèle**, même pente, $w$ toujours orthogonal.

### ✏️ Exercice 3 · Entraîner un perceptron à la main (le plus important de la séance)

> | point | $x_1$ | $x_2$ | $y$ |
> | --- | --- | --- | --- |
> | A | 1 | 3 | 1 |
> | B | 1 | 0 | 0 |
> | C | 3 | 1 | 1 |
> | D | 0 | 0 | 0 |
>
> Départ : $w = (0,0)$, $b = 0$, $\eta = 1$. Visite A, B, C, D dans cet ordre. Rappel : $z = 0$ donne $\hat y = 1$.
> 1. Tableau : point, $z$, $\hat y$, erreur, nouveau $w$, nouveau $b$.
> 2. Epochs jusqu'à une epoch sans erreur. Combien d'erreurs par epoch ?
> 3. La droite finale.

**Recette** (pour chaque ligne, dans l'ordre) :
1. $z = w_1x_1 + w_2x_2 + b$ avec les poids **actuels**.
2. $\hat y = 1$ si $z \ge 0$.
3. $e = y - \hat y$.
4. Si $e \neq 0$ : $w \leftarrow w + e\,x$ et $b \leftarrow b + e$. Sinon on ne touche à rien.

**Correction**

**Epoch 1**

| Point | $z$ | $\hat y$ | $e$ | nouveau $w$ | nouveau $b$ |
| --- | --- | --- | --- | --- | --- |
| A (1,3), y=1 | $0+0+0 = 0$ | 1 | 0 | (0, 0) | 0 |
| B (1,0), y=0 | $0$ | 1 | **−1** | $(0,0) - (1,0) = (-1, 0)$ | −1 |
| C (3,1), y=1 | $-3 + 0 - 1 = -4$ | 0 | **+1** | $(-1,0) + (3,1) = (2, 1)$ | 0 |
| D (0,0), y=0 | $0$ | 1 | **−1** | $(2,1) - (0,0) = (2, 1)$ | −1 |

→ **3 erreurs**. Remarque : pour D, $x = (0,0)$ donc **seul $b$ bouge**.

**Epoch 2**

| Point | $z$ | $\hat y$ | $e$ | nouveau $w$ | nouveau $b$ |
| --- | --- | --- | --- | --- | --- |
| A | $2 + 3 - 1 = 4$ | 1 | 0 | (2, 1) | −1 |
| B | $2 + 0 - 1 = 1$ | 1 | **−1** | $(2,1) - (1,0) = (1, 1)$ | −2 |
| C | $3 + 1 - 2 = 2$ | 1 | 0 | (1, 1) | −2 |
| D | $0 + 0 - 2 = -2$ | 0 | 0 | (1, 1) | −2 |

→ **1 erreur**.

**Epoch 3** : A : $1+3-2 = 2$ ✅ ; B : $1-2 = -1$ ✅ ; C : $3+1-2 = 2$ ✅ ; D : $-2$ ✅ → **0 erreur, on s'arrête**.

Erreurs par epoch : **3, 1, 0**.

**Droite finale** : $w = (1, 1)$, $b = -2$ → $x_1 + x_2 - 2 = 0$, soit $x_2 = 2 - x_1$. Intersections $(2, 0)$ et $(0, 2)$. A et C sont au-dessus (classe 1), B et D en dessous (classe 0). C'est exactement ce que renvoie `train_perceptron(X, y)` dans le cours : `w = [1., 1.], b = -2.0`.

### ✏️ Exercice 4 · XOR : aucune droite ne marche

> 1. Écris les 4 conditions sur $w_1, w_2, b$ pour qu'un perceptron calcule XOR.
> 2. Additionne les conditions de $(1,0)$ et $(0,1)$. Combine avec celle de $(0,0)$.
> 3. Conclus.

**Correction**

1. XOR vaut 1 quand **exactement une** entrée vaut 1.

| Entrée | Sortie voulue | Condition |
| --- | --- | --- |
| (0,0) | 0 | $b < 0$ |
| (0,1) | 1 | $w_2 + b \ge 0$ |
| (1,0) | 1 | $w_1 + b \ge 0$ |
| (1,1) | 0 | $w_1 + w_2 + b < 0$ |

2. Somme des lignes 2 et 3 : $w_1 + w_2 + 2b \ge 0$, donc $w_1 + w_2 + b \ge -b$. Or $b < 0$ donc $-b > 0$, d'où $w_1 + w_2 + b > 0$.
3. Ça **contredit** la ligne 4 ($w_1 + w_2 + b < 0$). **Aucun perceptron ne calcule XOR** : les données ne sont pas linéairement séparables.

### ⚠️ Pièges QCM · Séance 1

- $z = 0$ → $\hat y = 1$.
- $w$ est **orthogonal** à la frontière (pas parallèle). $b$ **translate**, il ne tourne pas.
- Novikoff : convergence **seulement si séparable**. Sur des données non séparables, la droite "saute" sans fin.
- La dérivée de la marche est **0 partout** (non définie en 0) → pas de descente de gradient possible.
- Une **epoch** = tout le dataset ; un **pas** = un exemple (ici).

---

## Séance 2 · Le neurone sigmoïde (régression logistique)

### 🧒 L'histoire

Le videur de la séance 1 répond juste "oui" ou "non". On le remplace par un videur qui dit **"je suis sûr à 90 %"**. Pour ça, on remplace la **marche d'escalier** (0 puis d'un coup 1) par une **rampe douce** : la sigmoïde. Comme la rampe a une **pente** partout, on peut savoir dans quel sens bouger les poids : c'est la **descente de gradient**.

Pour juger le modèle, on utilise la **log loss** : elle punit **très fort** quand le modèle est **sûr de lui et se trompe**.

### 📌 Ce qu'il faut savoir

**La sigmoïde**

$$a = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = w^\top x + b$$

- $0 < \sigma(z) < 1$ ; $\sigma(0) = 0.5$ ; $z \to +\infty$ : $\sigma \to 1$ ; $z \to -\infty$ : $\sigma \to 0$.
- **Symétrie** : $\sigma(-z) = 1 - \sigma(z)$.
- **Interprétation** : $a = P(y = 1 \mid x)$. Ce modèle s'appelle la **régression logistique** (un neurone + sigmoïde).
- **Décision** : classe 1 si $a \ge 0.5 \iff z \ge 0$. La frontière est **toujours la droite** $w^\top x + b = 0$ ! Ce qui change : chaque point a maintenant une **probabilité**.

**De la vraisemblance à la log loss**

- Un exemple : $P(y \mid x) = a^y (1-a)^{1-y}$ (Bernoulli : vaut $a$ si $y=1$, $1-a$ si $y=0$).
- Dataset : vraisemblance $L = \prod_i (a^{(i)})^{y^{(i)}} (1-a^{(i)})^{1-y^{(i)}}$. On veut la **maximiser**.
- On prend le log (même maximum, et évite qu'un produit de nombres < 1 tombe à 0), on met un moins, on divise par $m$ :

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)})\Big]$$

- Vocabulaire : **log loss** = **binary cross-entropy**. $L$ = loss d'**un** exemple, $J$ = **coût** = moyenne des $L$.
- Lecture : si $y = 1$, $L = -\log a$ (petit si $a \approx 1$, énorme si $a \approx 0$). Si $y = 0$, $L = -\log(1-a)$.

**Pourquoi pas l'erreur quadratique (MSE) ?**
1. **Pas convexe** en $w$ avec une sigmoïde → zones plates où la descente rampe.
2. **Gradient faible** : $\partial/\partial z\,(a-y)^2 = 2(a-y)\,a(1-a)$. Quand le modèle est très faux ($a \approx 0$, $y = 1$), $a(1-a) \approx 0$ → presque aucune correction.
La log loss, elle, est **convexe** et son gradient n'a **pas** ce facteur $a(1-a)$.

**La descente de gradient**

$$w \leftarrow w - \eta\frac{\partial J}{\partial w}, \qquad b \leftarrow b - \eta\frac{\partial J}{\partial b}$$

Le gradient pointe vers le haut, donc on va dans le sens **opposé**. $\eta$ trop petit → lent ; trop grand → on dépasse.

**Les 3 étapes du gradient (à savoir refaire)**

1. Dérivée de la sigmoïde : $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, soit $\dfrac{\partial a}{\partial z} = a(1-a)$. **Maximum en $z = 0$ : $0.5 \times 0.5 = 0.25$** (à retenir pour la séance 5).
2. Chain rule sur un exemple : $\dfrac{\partial L}{\partial a} = \dfrac{a - y}{a(1-a)}$, puis $\dfrac{\partial L}{\partial w_j} = \dfrac{a-y}{a(1-a)} \cdot a(1-a) \cdot x_j = (a - y)\,x_j$ et $\dfrac{\partial L}{\partial b} = a - y$.
3. Vectorisé sur le dataset ($X$ de shape $(m, n)$) :

$$dw = \frac{1}{m}X^\top(A - y) \quad\text{shape } (n,m)\times(m,) = (n,) \qquad db = \frac{1}{m}\sum_i (a^{(i)} - y^{(i)})$$

> **À retenir : le gradient = erreur × entrée, $(a - y)\,x$.** Même formule que la régression linéaire, avec $a = \sigma(z)$ au lieu de $z$.

**PyTorch**
- `requires_grad=True` : PyTorch enregistre les opérations. `loss.backward()` applique la chain rule à l'envers et remplit `w.grad`.
- **`BCEWithLogitsLoss`** : on lui donne le **logit $z$**, pas $\sigma(z)$. Raison : en float32, $\sigma(-200)$ arrondit à 0 et $\log 0 = -\infty$. La loss calcule une forme stable, $L = (1-y)z + \log(1 + e^{-z})$, sans jamais calculer $\sigma(z)$ seul. Exemple : $z = -200$, $y = 1$ → loss = 200 (fini et correct).
- La boucle en 5 lignes : `zero_grad`, forward, loss, `backward`, `step`.

**La limite** : la frontière reste une **droite**. Sur les "moons", ~0.88 de précision ; XOR toujours impossible.

### ✏️ Exercice 1 · La sigmoïde et un forward

> 1. Calcule $\sigma(0), \sigma(2), \sigma(-2), \sigma(4)$ avec $e^{-2} \approx 0.135$, $e^{-4} \approx 0.018$. Vérifie $\sigma(-2) = 1 - \sigma(2)$.
> 2. Un neurone a $w = (2, -1)$, $b = -1$. Pour $x = (1, 3)$, calcule $z$ puis $a$.
> 3. Que vaut $P(y=1 \mid x)$ ? Quelle classe ?

**Correction**

1. - $\sigma(0) = 1/(1+1) = 0.5$
   - $\sigma(2) = 1/(1 + 0.135) = 0.881$
   - $\sigma(-2) = 1/(1 + e^{2}) = 1/(1 + 7.389) = 0.119$ ; et $1 - 0.881 = 0.119$ ✅ (symétrie)
   - $\sigma(4) = 1/(1 + 0.018) = 0.982$
2. $z = 2 \times 1 + (-1) \times 3 + (-1) = 2 - 3 - 1 = -2$, donc $a = \sigma(-2) = 0.119$.
3. $P(y = 1 \mid x) = 0.119$ (11.9 %) < 0.5 → **classe 0**.

### ✏️ Exercice 2 · Trois log loss à la main

> Trois exemples de vraie classe $y = 1$, prédictions $a = 0.9$, $0.5$, $0.1$.
> 1. Loss de chacun (log naturel). 2. Coût $J$. 3. Qui contribue le plus ? Que devient sa loss si $a = 0.01$ ?

**Recette** : $y = 1$ → seul le terme $-\ln a$ est actif.

**Correction**

1. $-\ln 0.9 = 0.105$ ; $-\ln 0.5 = 0.693$ ; $-\ln 0.1 = 2.303$.
2. $J = (0.105 + 0.693 + 2.303)/3 = 3.101/3 = 1.034$.
3. Le 3e ($a = 0.1$) : 2.303 sur 3.101, environ **74 %** du total. Avec $a = 0.01$ : $-\ln 0.01 = 4.605$. Plus le modèle est **sûr de lui et faux**, plus ça coûte, **sans limite** ($-\ln a \to \infty$ quand $a \to 0$).

### ✏️ Exercice 3 · Un pas de descente de gradient à la main

> | | $x_1$ | $x_2$ | $y$ |
> | --- | --- | --- | --- |
> | exemple 1 | 2 | 1 | 1 |
> | exemple 2 | 0 | 2 | 0 |
>
> Départ $w = (0, 0)$, $b = 0$, $\eta = 2$. On donne $e^{-1.5} \approx 0.223$, $e^{1} \approx 2.718$.
> 1. Forward : $a^{(1)}, a^{(2)}$, coût $J$. 2. $dw = \frac{1}{2}X^\top(A - y)$ et $db$. 3. Mise à jour. 4. Nouveau forward, nouveau coût : $J$ a-t-il baissé ?

**Recette** : forward → erreurs $A - y$ → gradients → $w - \eta\,dw$ → on recommence le forward.

**Correction**

1. **Forward** : $w = 0, b = 0$ donc $z = 0$ pour les deux, $a^{(1)} = a^{(2)} = 0.5$.
   $J = -\frac{1}{2}[\ln 0.5 + \ln(1 - 0.5)] = -\ln 0.5 = \ln 2 = 0.693$.
2. **Gradients** : $A - y = (0.5 - 1,\ 0.5 - 0) = (-0.5,\ +0.5)$.
   $X^\top = \begin{pmatrix}2 & 0\\ 1 & 2\end{pmatrix}$, donc $X^\top(A - y) = (2 \times (-0.5) + 0 \times 0.5,\ \ 1 \times (-0.5) + 2 \times 0.5) = (-1,\ 0.5)$.
   $dw = \frac{1}{2}(-1, 0.5) = (-0.5,\ 0.25)$ ; $db = \frac{1}{2}(-0.5 + 0.5) = 0$.
3. **Mise à jour** : $w = (0,0) - 2 \times (-0.5, 0.25) = (1,\ -0.5)$ ; $b = 0 - 2 \times 0 = 0$.
4. **Nouveau forward** :
   - $z^{(1)} = 1 \times 2 + (-0.5) \times 1 = 1.5$ → $a^{(1)} = 1/(1 + 0.223) = 0.818$
   - $z^{(2)} = 1 \times 0 + (-0.5) \times 2 = -1$ → $a^{(2)} = 1/(1 + 2.718) = 0.269$
   - $J = -\frac{1}{2}[\ln 0.818 + \ln(1 - 0.269)] = -\frac{1}{2}[-0.201 - 0.313] = 0.257$

   **Oui**, $J$ passe de **0.693 à 0.257**. L'exemple 1 (classe 1) a monté à 0.818, l'exemple 2 (classe 0) est descendu à 0.269 : les deux vont dans le bon sens.

### ⚠️ Pièges QCM · Séance 2

- La frontière de la régression logistique est **linéaire**.
- $\max \sigma'(z) = 0.25$ (en $z = 0$), pas 0.5 ni 1.
- Gradient de la log loss avec sigmoïde : $(a - y)x$, **sans** facteur $a(1-a)$.
- Minimiser la log loss = maximiser la vraisemblance.
- `BCEWithLogitsLoss` prend des **logits** : ne **pas** mettre de `sigmoid` avant.
- Le code du cours : `log_loss` ajoute `eps = 1e-15` pour éviter $\log(0)$.

---

## Séance 3 · Premier réseau et backpropagation

### 🧒 L'histoire

Un neurone = **une** règle = **une** droite. Pour XOR il faut **deux** règles : "au moins un des deux" (OR) **et** "pas les deux" (NAND). Un réseau, c'est une **équipe** : les neurones cachés posent des **sous-questions**, le neurone de sortie **combine** leurs réponses.

Pour entraîner l'équipe, on remonte la chaîne des responsabilités depuis la fin : *"la sortie s'est trompée de tant ; à quel point est-ce la faute de chaque neurone caché ? et de chaque poids ?"*. C'est la **backpropagation** : la chain rule appliquée de la fin vers le début.

### 📌 Ce qu'il faut savoir

- $\text{XOR}(x_1, x_2) = \text{AND}(\text{OR}(x_1,x_2),\ \text{NAND}(x_1,x_2))$ : XOR vaut 1 **entre** les deux droites.
- La **couche cachée** crée une **nouvelle représentation** $h = (h_1, h_2)$ dans laquelle le problème devient séparable par une droite.

| Entrée $x$ | Caché $h$ | XOR |
| --- | --- | --- |
| (0, 0) | (0, 1) | 0 |
| (0, 1), (1, 0) | (1, 1) | 1 |
| (1, 1) | (1, 0) | 0 |

**Vocabulaire et shapes**
- Couche $l$ a $n^{[l]}$ neurones ; couche 0 = entrée ($n^{[0]} = n$ features). Réseau "2-3-1" = 2 entrées, 3 cachés, 1 sortie.
- $W^{[l]}$ de shape $(n^{[l]}, n^{[l-1]})$ (une **ligne** par neurone), $b^{[l]}$ de shape $(n^{[l]},)$.
- **Nombre de paramètres de la couche $l$** : $n^{[l]} \times n^{[l-1]} + n^{[l]} = n^{[l]}(n^{[l-1]} + 1)$.

**Forward** (un exemple en colonne) : $z^{[1]} = W^{[1]}x + b^{[1]}$, $a^{[1]} = \sigma(z^{[1]})$, $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$, $a^{[2]} = \sigma(z^{[2]})$.

**En code** (NumPy / PyTorch mettent un exemple **par ligne**) : $Z^{[l]} = A^{[l-1]}W^{[l]\top} + b^{[l]}$, shape $(m, n^{[l]})$. C'est exactement `nn.Linear(n_in, n_out)` : `X @ W.T + b`, avec `weight` de shape **`(n_out, n_in)`**.

**Pourquoi une activation non linéaire ?** Sans elle : $W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]} = W'x + b'$ → **une seule couche linéaire**, une seule droite, quel que soit le nombre de couches. Oublier l'activation entre deux `nn.Linear` est un **bug classique** : le code tourne, mais le modèle est linéaire.

**Backpropagation** (Rumelhart, Hinton, Williams, 1986). Notation $dz = \partial L/\partial z$.

| Couche | Formule (vectorisée sur $m$ exemples en colonnes) |
| --- | --- |
| sortie | $dZ^{[2]} = A^{[2]} - Y$ |
| | $dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]\top}$, $\quad db^{[2]} = \frac{1}{m}\sum_{\text{colonnes}} dZ^{[2]}$ |
| cachée | $dZ^{[1]} = W^{[2]\top}dZ^{[2]} \odot A^{[1]} \odot (1 - A^{[1]})$ |
| | $dW^{[1]} = \frac{1}{m}dZ^{[1]}X^\top$, $\quad db^{[1]} = \frac{1}{m}\sum_{\text{colonnes}} dZ^{[1]}$ |

**Règle générale** : $dZ^{[l]} = W^{[l+1]\top}dZ^{[l+1]} \odot \sigma'(Z^{[l]})$ et $dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]\top}$.

- $\odot$ = produit **élément par élément**.
- $W^{[2]\top}$ **renvoie l'erreur** de la sortie vers chaque neurone caché, pondérée par la connexion utilisée à l'aller.
- **Contrôle** : chaque gradient a **la même shape que son paramètre**.

**PyTorch** : autograd fait exactement ce calcul (`loss.backward()`). `nn.Sequential(nn.Linear(2,3), nn.Sigmoid(), nn.Linear(3,1))` = réseau 2-3-1 (13 paramètres), **pas de sigmoïde à la fin** car `BCEWithLogitsLoss` l'applique. Sur les moons : régression logistique 0.885, réseau 2-3-1 **1.000** (en train ; la généralisation demande un test set).

### ✏️ Exercice 1 · Le réseau XOR à la main

> Activation marche. $h_1$ = OR : $w = (1,1)$, $b = -0.5$. $h_2$ = NAND : $w = (-1,-1)$, $b = 1.5$. Sortie $\hat y$ = AND de $(h_1, h_2)$ : $w = (1,1)$, $b = -1.5$.
> 1. Calcule $h_1, h_2, \hat y$ pour les 4 entrées. Est-ce XOR ?
> 2. Remplace chaque marche par une sigmoïde et multiplie tous les poids et biais par 10. Calcule $h_1, h_2, \hat y$ pour $x = (0,1)$. ($\sigma(5) \approx 0.993$, $\sigma(4.87) \approx 0.992$.)

**Correction**

1. Les 4 entrées, ligne par ligne :

| $x$ | $z_1 = x_1+x_2-0.5$ | $h_1$ | $z_2 = -x_1-x_2+1.5$ | $h_2$ | $h_1+h_2-1.5$ | $\hat y$ |
| --- | --- | --- | --- | --- | --- | --- |
| (0,0) | −0.5 | 0 | 1.5 | 1 | −0.5 | 0 |
| (0,1) | 0.5 | 1 | 0.5 | 1 | 0.5 | 1 |
| (1,0) | 0.5 | 1 | 0.5 | 1 | 0.5 | 1 |
| (1,1) | 1.5 | 1 | −0.5 | 0 | −0.5 | 0 |

$\hat y = (0, 1, 1, 0)$ : **c'est XOR**. ✅

2. $x = (0,1)$, poids × 10 :
   - $z_1 = 10(0 + 1) - 5 = 5$ → $h_1 = \sigma(5) = 0.993$
   - $z_2 = -10(0 + 1) + 15 = 5$ → $h_2 = \sigma(5) = 0.993$
   - $z = 10(0.993 + 0.993) - 15 = 4.87$ → $\hat y = \sigma(4.87) = 0.992$ → classe 1 ✅

   Multiplier par 10 rend la sigmoïde **très raide** : elle ressemble à la marche, mais elle est **dérivable**.

### ✏️ Exercice 2 · Shapes et nombre de paramètres

> 1. Réseau 2-3-1 : shapes de $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}$ et total.
> 2. Réseau 784-128-10 (images 28×28). Mêmes questions.
> 3. Quelle couche a le plus de paramètres et pourquoi ?

**Recette** : $W^{[l]}$ = (neurones de la couche, neurones de la couche d'avant). Paramètres = $n_\text{out} \times n_\text{in} + n_\text{out}$.

**Correction**

1. $W^{[1]}$ : (3, 2), $b^{[1]}$ : (3,), $W^{[2]}$ : (1, 3), $b^{[2]}$ : (1,). Total : $6 + 3 + 3 + 1 = $ **13**.
2. $W^{[1]}$ : (128, 784), $b^{[1]}$ : (128,), $W^{[2]}$ : (10, 128), $b^{[2]}$ : (10,).
   Couche 1 : $784 \times 128 + 128 = 100\,480$. Couche 2 : $128 \times 10 + 10 = 1\,290$. Total : **101 770**.
3. La **première couche** (98.7 % des paramètres) : chacun des 128 neurones est relié à **chacun des 784 pixels**.

### ✏️ Exercice 3 · Forward et backward à la main (incontournable)

> Réseau 2-2-1 avec sigmoïdes, un exemple $x = (1, 2)$, label $y = 1$ :
> $W^{[1]} = \begin{pmatrix}1 & -1\\ 0.5 & 0.5\end{pmatrix}$, $b^{[1]} = \begin{pmatrix}1\\ -1.5\end{pmatrix}$, $W^{[2]} = \begin{pmatrix}1 & -1\end{pmatrix}$, $b^{[2]} = 0$.
> 1. Forward : $z^{[1]}, a^{[1]}, z^{[2]}, a^{[2]}$ et $L$.
> 2. Backward : $dz^{[2]}, dW^{[2]}, db^{[2]}, da^{[1]}, dz^{[1]}, dW^{[1]}, db^{[1]}$.
> 3. Un pas avec $\eta = 1$ : nouveaux $W^{[2]}_1$ et $W^{[1]}_{11}$.
> 4. Bonus : mettre à jour tous les paramètres et recalculer $a^{[2]}$ ($\sigma(0.75) \approx 0.679$, $\sigma(1.108) \approx 0.752$).

**Recette** :
1. Forward couche par couche, en gardant toutes les valeurs.
2. Backward : commence par $dz^{[2]} = a^{[2]} - y$.
3. "Gradient d'un poids = erreur du neurone × ce qui entre dans le poids".
4. Pour passer à la couche d'avant : $da^{[1]} = W^{[2]\top}dz^{[2]}$, puis on multiplie par $\sigma' = a(1-a)$.

**Correction**

**1. Forward**
- $z^{[1]} = W^{[1]}x + b^{[1]} = (1 \times 1 - 1 \times 2 + 1,\ \ 0.5 \times 1 + 0.5 \times 2 - 1.5) = (0,\ 0)$
- $a^{[1]} = (\sigma(0), \sigma(0)) = (0.5,\ 0.5)$
- $z^{[2]} = 1 \times 0.5 - 1 \times 0.5 + 0 = 0$ → $a^{[2]} = 0.5$
- $L = -\ln 0.5 = 0.693$

**2. Backward**
- $dz^{[2]} = a^{[2]} - y = 0.5 - 1 = -0.5$
- $dW^{[2]} = dz^{[2]} \cdot a^{[1]\top} = -0.5 \times (0.5, 0.5) = (-0.25,\ -0.25)$
- $db^{[2]} = -0.5$
- $da^{[1]} = W^{[2]\top}dz^{[2]} = (1, -1) \times (-0.5) = (-0.5,\ 0.5)$
- $dz^{[1]} = da^{[1]} \odot a^{[1]}(1 - a^{[1]}) = (-0.5 \times 0.25,\ 0.5 \times 0.25) = (-0.125,\ 0.125)$
- $dW^{[1]} = dz^{[1]}x^\top = \begin{pmatrix}-0.125 \times 1 & -0.125 \times 2\\ 0.125 \times 1 & 0.125 \times 2\end{pmatrix} = \begin{pmatrix}-0.125 & -0.25\\ 0.125 & 0.25\end{pmatrix}$
- $db^{[1]} = (-0.125,\ 0.125)$

Ce sont **exactement** les valeurs que donne `W2.grad` et `W1.grad` avec autograd dans le cours.

**3. Un pas** ($\eta = 1$) : $W^{[2]}_1 = 1 - 1 \times (-0.25) = 1.25$ ; $W^{[1]}_{11} = 1 - 1 \times (-0.125) = 1.125$.

**4. Bonus** : tous les paramètres
- $W^{[1]} = \begin{pmatrix}1.125 & -0.75\\ 0.375 & 0.25\end{pmatrix}$, $b^{[1]} = (1.125,\ -1.625)$, $W^{[2]} = (1.25,\ -0.75)$, $b^{[2]} = 0.5$
- $z^{[1]} = (1.125 - 1.5 + 1.125,\ \ 0.375 + 0.5 - 1.625) = (0.75,\ -0.75)$ → $a^{[1]} = (0.679,\ 0.321)$
- $z^{[2]} = 1.25 \times 0.679 - 0.75 \times 0.321 + 0.5 = 1.108$ → $a^{[2]} = 0.752$

$a^{[2]}$ passe de 0.5 à **0.752** (la loss passe de 0.693 à 0.285) : le réseau s'approche du label $y = 1$. ✅

### ⚠️ Pièges QCM · Séance 3

- `nn.Linear(3, 5).weight.shape` = **(5, 3)** (sortie, entrée).
- Deux couches linéaires sans activation = **une** couche linéaire.
- La première couche dense contient presque tous les paramètres.
- La backprop part **de la fin** et **réutilise** les résultats de l'étape précédente.
- Une précision de 1.000 **sur le train** ne dit rien sur la généralisation.

---

## Séance 4 · PyTorch en profondeur

### 🧒 L'histoire

PyTorch, c'est une **boîte à outils** avec 6 outils :

| Outil | Rôle "pour un enfant" | Dans la boucle |
| --- | --- | --- |
| **Tensor** | le tableau de nombres (sur CPU ou GPU) | `X`, `y`, les poids |
| **Autograd** | le calculateur automatique de pentes | `loss.backward()` |
| **nn.Module** | le plan du modèle | `model(X)` |
| **Loss** | le juge qui donne une note d'erreur | `loss_fn(logits, y)` |
| **Optimiseur** | celui qui corrige les poids | `optimizer.step()` |
| **DataLoader** | le serveur qui apporte les données par petits plateaux mélangés | `for xb, yb in loader` |

Données du jour : *digits* de scikit-learn, 1797 images 8×8, 10 classes.

### 📌 Ce qu'il faut savoir

**Tenseurs**
- `shape` (la plupart des bugs sont des bugs de shape : **imprime-la**), `dtype` (**float32** pour données et poids, **int64** pour les labels de classe), `device` (`cpu` ou `cuda` ; tous les tenseurs d'une opération doivent être sur le même).
- `@` = produit matriciel ; `* + -` = élément par élément. `X.sum(dim=0)` réduit sur les lignes (un résultat **par colonne**). `.item()` = tenseur à 1 élément → nombre Python.
- **Broadcasting** : on compare les shapes **depuis la droite** ; deux tailles sont compatibles si **égales** ou si l'une vaut **1** (ou manque).
  - `(2,3) + (3,)` ✅ : ajouté à chaque ligne (c'est le `+ b` d'une couche).
  - `(2,3) + (2,1)` ✅ : ajouté à chaque colonne.
  - `(2,3) + (2,)` ❌ : erreur (3 ≠ 2).

**Autograd**
- `requires_grad=True` = "je voudrai le gradient par rapport à ce tenseur". Chaque résultat garde un `grad_fn`.
- Exemple : `x = 2`, `f = x**2 + 3*x` = 10 ; `f.backward()` → `x.grad` = $2x + 3$ = **7**.
- **Les gradients s'accumulent** : `backward()` **ajoute** à `.grad`. Deux `backward()` de $(3w)^2$ avec $w = 1$ → 18 + 18 = **36**. D'où `optimizer.zero_grad()` à chaque pas.
- `with torch.no_grad():` → rien n'est enregistré (pour la mise à jour manuelle et l'évaluation). `t.detach()` = mêmes valeurs, coupé du graphe.

**nn.Module**
- `__init__` déclare les couches (`super().__init__()` **en premier**), `forward` les utilise. On appelle `model(x)`, jamais `model.forward(x)`.
- MLP 64-32-10 : $64 \times 32 + 32 + 32 \times 10 + 10 = $ **2410** paramètres.

**Dataset et DataLoader**
- `Dataset` : `__len__` (combien d'exemples) et `__getitem__` (l'exemple $i$).
- `DataLoader(ds, batch_size=32, shuffle=True)` : `shuffle=True` pour le train, `False` pour validation et test.
- 100 exemples, batchs de 32 → 32, 32, 32, 4 → **4 batchs**.

**Mini-batch gradient descent**
- Un **pas** (itération) = un batch, une mise à jour. Une **epoch** = un passage sur le train = $\lceil m/B \rceil$ pas. Digits : 1197 images, $B = 32$ → **38 pas** par epoch.
- Le gradient d'un batch est une estimation **bruitée** du vrai gradient, mais $m/B$ fois moins chère.

**Softmax** (de 2 à $C$ classes)

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}, \qquad p_k > 0, \quad \sum_k p_k = 1$$

- Garde l'ordre (plus grand logit → plus grande proba) ; prédiction = $\arg\max_k z_k$.
- **Amplifie les écarts** : un logit plus grand de 1 → $e \approx 2.7$ fois plus probable.
- **Ajouter une constante à tous les logits ne change rien** (seules les différences comptent ; les librairies soustraient $\max_j z_j$ pour éviter l'overflow).
- Avec $C = 2$ : $p_1 = \sigma(z_1 - z_0)$ → c'est la sigmoïde.

**Cross-entropy**

$$L = -\sum_{k} y_k \log p_k = -\log p_\text{vraie classe}, \qquad \frac{\partial L}{\partial z} = p - y$$

- Seule la proba de la **vraie** classe compte. $p_\text{vraie} = 1$ → loss 0 ; $p_\text{vraie} \to 0$ → loss $\to \infty$.
- Même forme que la séance 2 : **prédiction moins vérité**. Pour $C = 2$, c'est la log loss binaire.
- Réseau qui ne sait rien → $p_k = 1/C$ → loss $= \ln C$ ($\ln 10 = 2.30$).

**`nn.CrossEntropyLoss`** — deux erreurs classiques :
1. **Pas de softmax avant** : elle applique log-softmax elle-même. On lui donne les **logits**.
2. **Labels** = indices de classe, shape `(B,)`, dtype **int64**. Pas de one-hot, pas de float.
- Classes numérotées **à partir de 0** en code. Le gradient vaut $(p - y)/B$ car la loss est une **moyenne** sur le batch.

**Optimiseur** : `SGD(model.parameters(), lr)` ; `step()` fait `p -= lr * p.grad` pour chaque paramètre. Il **ne calcule pas** les gradients (c'est `backward()`). `Adam(params, lr=1e-3)` adapte le pas de chaque paramètre.

**La boucle d'entraînement — les 5 lignes à connaître par cœur**

```python
for epoch in range(n_epochs):
    model.train()                          # mode entraînement (dropout, batch norm)
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                 # 1. forward
        loss = loss_fn(logits, yb)         # 2. loss
        optimizer.zero_grad()              # 3. remise à zéro des gradients
        loss.backward()                    # 4. backpropagation
        optimizer.step()                   # 5. mise à jour
    val_loss, val_acc = evaluate(model, val_loader, loss_fn)
```

**La boucle d'évaluation** : `model.eval()` + `torch.no_grad()` ; prédiction = `logits.argmax(dim=1)` (pas besoin de softmax) ; on pondère chaque loss de batch par `len(yb)` car le dernier batch peut être plus petit.

**Train / validation / test**
- **Train** (1197) : calculer les gradients, mettre à jour.
- **Validation** (300) : **choisir** (epochs, learning rate, architecture).
- **Test** (300) : rapporter le score final, **une seule fois**, à la fin.
- Si on choisit le meilleur de 20 réglages sur un set, son score devient optimiste → on choisit sur la validation.
- Split avec une **seed fixe** pour qu'il soit reproductible.

**Sauvegarde** : `torch.save(model.state_dict(), "f.pt")` (les nombres seulement : il faut la classe pour reconstruire), `model2.load_state_dict(torch.load("f.pt"))`. Modèle et données sur le même device, sinon `RuntimeError: Expected all tensors to be on the same device`.

### ✏️ Exercice 1 · Une softmax à la main

> Logits $z = (2, 1, 0)$ pour (chat, chien, oiseau). $e^2 \approx 7.389$, $e^1 \approx 2.718$.
> 1. Calcule $p_1, p_2, p_3$, vérifie la somme. 2. Classe prédite ? Combien de fois plus probable que la 2e ?
> 3. Logits $(12, 11, 10)$ ? Pourquoi ? 4. Logits $(4, 2, 0)$ : plus ou moins confiant ?

**Recette** : exponentielle de chaque logit → somme → divise chaque exponentielle par la somme.

**Correction**

1. $e^2 + e^1 + e^0 = 7.389 + 2.718 + 1 = 11.107$.
   $p = (7.389/11.107,\ 2.718/11.107,\ 1/11.107) = (0.665,\ 0.245,\ 0.090)$. Somme = 1.000 ✅
2. **Chat**. Rapport $p_1/p_2 = e^{2-1} = e \approx$ **2.72 fois** plus probable.
3. **Exactement les mêmes probas** $(0.665, 0.245, 0.090)$ : on a ajouté 10 à tous les logits, et seules les **différences** comptent.
4. $e^4 \approx 54.6$, $e^2 = 7.389$, $e^0 = 1$ ; somme 62.99 → $p = (0.867,\ 0.117,\ 0.016)$. **Plus confiant** : multiplier les logits par 2 creuse les écarts.

### ✏️ Exercice 2 · Cross-entropy à la main

> $p = (0.665, 0.245, 0.090)$. $\ln 0.665 \approx -0.408$, $\ln 0.090 \approx -2.408$, $\ln 10 \approx 2.303$.
> 1. Vraie classe chat : $L$ ? 2. Vraie classe oiseau : $L$ ? Compare.
> 3. Vraie classe chat : gradient $\partial L/\partial z$. Quels logits monteront, lesquels descendront ?
> 4. Réseau non entraîné à 10 classes ($p_k \approx 1/10$) : loss attendue ?

**Correction**

1. $L = -\ln 0.665 = 0.408$.
2. $L = -\ln 0.090 = 2.408$ : presque **6 fois plus** grand, car le modèle donnait peu de proba à la vraie classe.
3. $y = (1, 0, 0)$ donc $p - y = (0.665 - 1,\ 0.245,\ 0.090) = (-0.335,\ 0.245,\ 0.090)$.
   La descente fait $z \leftarrow z - \eta(p - y)$ : le logit **chat monte** (gradient négatif), **chien et oiseau descendent** (gradient positif).
4. $L = -\ln(1/10) = \ln 10 = $ **2.303**. C'est le **contrôle de départ** : si ta loss initiale est loin de 2.30, il y a un bug.

> Vérif avec le code du cours : deux exemples de logits $(2,1,0)$ avec labels 0 et 2 → loss moyenne $(0.408 + 2.408)/2 = 1.4076$.

### ✏️ Exercice 3 · Trouver les cinq bugs

```python
 1  model = MLP()
 2  loss_fn = nn.CrossEntropyLoss()
 3  optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
 4  for epoch in range(30):
 5      for xb, yb in train_loader:
 6          probs = torch.softmax(model(xb), dim=1)
 7          loss = loss_fn(probs, yb.float())
 8          loss.backward()
 9          optimizer.step()
10  correct = 0
11  with torch.no_grad():
12      for xb, yb in train_loader:
13          correct += (model(xb).argmax(dim=1) == yb).sum().item()
14  print("accuracy:", correct / len(train_loader.dataset))
```

> Indice : deux bugs dans la loss, une ligne manquante, deux dans l'évaluation.

**Correction**

| # | Ligne | Problème | Correctif |
| --- | --- | --- | --- |
| 1 | 6 | `softmax` avant `CrossEntropyLoss`, qui applique déjà log-softmax → double softmax, gradients écrasés | `logits = model(xb)` |
| 2 | 7 | `yb.float()` : la loss veut des **indices de classe int64** | `loss = loss_fn(logits, yb)` |
| 3 | entre 7 et 8 | **`optimizer.zero_grad()` manque** : les gradients s'accumulent de batch en batch | ajouter `optimizer.zero_grad()` avant `loss.backward()` |
| 4 | 10–11 | **`model.eval()` manque** avant l'évaluation (compte pour dropout et batch norm) ; et `model.train()` au début de chaque epoch | `model.eval()` avant la boucle d'éval |
| 5 | 12 (et 14) | on évalue sur **`train_loader`** : score optimiste, le modèle a vu ces images | `val_loader` (et le test une seule fois à la fin) |

### ⚠️ Pièges QCM · Séance 4

- `backward()` **additionne** dans `.grad` → `zero_grad()` obligatoire.
- `CrossEntropyLoss` : **logits** + labels **int64** de shape `(B,)`.
- Softmax invariante par **ajout** d'une constante, **pas** par multiplication.
- Loss initiale ≈ $\ln C$.
- `optimizer.step()` ne calcule pas les gradients.
- On **choisit** sur la validation, on **rapporte** sur le test, une fois.
- Nombre de pas par epoch = $\lceil m/B \rceil$ (on arrondit au-dessus).

---

## Séance 5 · Vision, activations, initialisation

### 🧒 L'histoire

**Le téléphone arabe.** Pendant la backprop, le message d'erreur part de la sortie et doit traverser toutes les couches. Chaque couche sigmoïde **divise le volume par au moins 4**. Après 6 couches, les premières couches n'entendent **plus rien** : elles n'apprennent pas. C'est le **vanishing gradient**.

**ReLU** = un relais qui transmet **à volume 1** (ou se tait complètement). **L'initialisation** = régler le volume de départ : trop bas → silence, trop fort → cris (**explosion**).

### 📌 Ce qu'il faut savoir

**Le projet** : FashionMNIST (Zalando, 2017), 70 000 images 28×28, 10 classes (60 000 train dont 10 000 pour la validation, 10 000 test). Plus dur que digits : chemise, T-shirt, pull, manteau se ressemblent.
- `nn.Flatten()` : (B, 28, 28) → (B, 784), ligne après ligne. **Perd la structure 2D** (corrigé en séance 7).
- Standardiser : $x \leftarrow (x - \mu)/s$, avec $\mu, s$ calculés **sur le train seulement**.
- 784-128-10 sigmoïde : **101 770** paramètres, ~**82 %** après 1 epoch, ~**86 %** après 5.

**Plus profond = pire** : 6 couches cachées sigmoïdes → **10 %** (le hasard), loss bloquée à $\ln 10 = 2.30$. Pas un bug : le **gradient** de la 1re couche est 20 000 fois plus petit que celui de la dernière ($3.6 \times 10^{-5}$ contre 0.77), divisé par ~7 à chaque couche.

**Pourquoi** : la backprop multiplie **une dérivée par couche** :

$$\frac{\partial a^{[L]}}{\partial a^{[0]}} = \prod_{l=1}^{L} w^{[l]}\,\sigma'(z^{[l]})$$

Facteurs < 1 → **vanishing** (décroissance exponentielle avec la profondeur). Facteurs > 1 → **exploding**.

**$\sigma'$ est au plus 1/4** : avec $s = \sigma(z)$, $g(s) = s(1-s)$ est maximal en $s = 1/2$ → $\max \sigma' = 1/4$ (en $z = 0$). Quand $|z|$ est grand, le neurone **sature** : $\sigma'(5) = 0.0066$.

**ReLU** : $\text{ReLU}(z) = \max(0, z)$, dérivée **1** si $z > 0$, **0** si $z < 0$.
- Sur un chemin actif, la dérivée vaut exactement 1 → le produit ne rétrécit pas.
- Ne sature pas pour $z > 0$ ; très peu chère (une comparaison) ; **toujours non linéaire** (le coude en 0).
- Nair et Hinton (2010), Glorot (2011), AlexNet (2012) : le défaut pour les couches cachées.

**Le prix : les neurones morts (dead ReLU)**. Si $z < 0$ pour **toutes** les entrées, le neurone sort 0 et sa dérivée est 0 partout → aucun gradient n'atteint $w$ et $b$ → **il ne revient jamais**. Causes : learning rate trop grand, mauvaise init. Correctifs : lr raisonnable, init He, **Leaky ReLU** $\max(0.01z, z)$.

**Les autres activations**

| Activation | Où | Pourquoi |
| --- | --- | --- |
| **ReLU** | couches cachées, choix par défaut (MLP, CNN) | dérivée 1, pas chère, marche |
| Leaky ReLU, ELU | couches cachées si beaucoup de neurones meurent | gradient jamais exactement nul |
| **GELU** $= z\Phi(z)$ | couches cachées des **transformers** (BERT, GPT) | ReLU lisse, un peu mieux dans les grands modèles |
| **tanh** | dans les cellules **RNN et LSTM** | bornée dans $(-1, 1)$, **centrée sur 0**, $\tanh'(0) = 1$, mais sature |
| **sigmoïde** | **sortie** 2 classes ou multi-label ; portes du LSTM | une proba dans $(0, 1)$ ; à éviter dans les piles profondes |
| **softmax** | **sortie** pour $C$ classes exclusives | $C$ probas qui somment à 1 (dans `CrossEntropyLoss` en PyTorch) |

> Sortie : choisie par la **tâche**. Couches cachées : choisies pour le **flux de gradient**.

**Initialisation : pourquoi pas tout à zéro ?** Si tous les poids d'une couche sont égaux : même valeur au forward, même gradient au backward, même mise à jour → **égaux pour toujours** (128 neurones identiques = 1 neurone). Avec des zéros exacts, c'est pire : le gradient des couches précédentes est nul. → Il faut **casser la symétrie** : poids **aléatoires centrés sur 0** (les biais peuvent partir de 0).

**Mais quelle taille ?** 10 couches ReLU de 256 :
- $\mathcal{N}(0, 0.01^2)$ : chaque couche multiplie le signal par ~0.11 → $4 \times 10^{-10}$ après 10 couches (**disparaît**).
- $\mathcal{N}(0, 1)$ : ~11 par couche → $4 \times 10^{10}$ (**explose**).

**L'argument de variance** : $z = \sum_{i=1}^{n_\text{in}} w_i x_i$ avec $w, x$ indépendants de moyenne 0 :

$$\text{Var}(z) = n_\text{in}\,\text{Var}(w)\,\text{Var}(x) \quad\Rightarrow\quad \text{pour garder Var}(z) = \text{Var}(x) : \ \text{Var}(w) = \frac{1}{n_\text{in}}$$

| Init | Variance des poids | Pour |
| --- | --- | --- |
| **Xavier / Glorot** (2010) | $\dfrac{1}{n_\text{in}}$ ou $\dfrac{2}{n_\text{in} + n_\text{out}}$ (moyenne harmonique forward/backward) | tanh, sigmoïde |
| **He / Kaiming** (2015) | $\dfrac{2}{n_\text{in}}$ (le 2 compense la ReLU qui met à 0 la moitié des valeurs) | **ReLU** |

- Une uniforme $U(-a, a)$ a pour variance $a^2/3$ ; `xavier_uniform_` utilise $a = \sqrt{6/(n_\text{in} + n_\text{out})}$.
- Défaut de `nn.Linear` en PyTorch : variante Kaiming-uniform avec $a = 1/\sqrt{n_\text{in}}$, donc $\text{Var}(w) = 1/(3n_\text{in})$ : le bon ordre, plus petit que He.
- Le `_` final (`kaiming_normal_`) = modification **en place**.

### ✏️ Le vote · « Rendre le réseau plus profond ? »

> Même données, même SGD, même lr, mais 6 couches cachées sigmoïdes de 128 au lieu de 1. Après 5 epochs : (a) mieux, (b) pareil, (c) moins bien, (d) environ 10 % ?

**Réponse : (d) ~10 %**, le niveau du hasard sur 10 classes. Le gradient disparaît dans les couches sigmoïdes : les premières couches restent aléatoires, les dernières n'ont rien d'utile à apprendre.

### ✏️ Exercice 1 · À quelle vitesse le gradient disparaît-il ?

> Chaîne de neurones sigmoïdes, un par couche, tous les poids $w^{[l]} = 1$.
> 1. Majorant de $|\partial a^{[5]}/\partial a^{[0]}|$ pour 5 couches, puis 10.
> 2. Avec $z = (0.5, -1, 2, 1, -0.5)$ et $\sigma(z) = (0.622, 0.269, 0.881, 0.731, 0.378)$ : les cinq $\sigma'(z^{[l]})$ et leur produit.
> 3. La dernière couche reçoit un gradient d'ordre 1 et $\eta = 0.1$ : ordre de la mise à jour du poids de la 1re couche ?
> 4. Peut-on corriger avec de grands poids, $w = 4$, pour que $w\sigma'(0) = 1$ ?

**Recette** : $\sigma' = \sigma(1-\sigma)$ ; le gradient de la chaîne = produit des $w \cdot \sigma'$.

**Correction**

1. Chaque facteur $\le 1 \times 1/4$. 5 couches : $(1/4)^5 = 1/1024 \approx 10^{-3}$. 10 couches : $(1/4)^{10} \approx 9.5 \times 10^{-7}$ (~$10^{-6}$).
2. $\sigma' = s(1-s)$ :
   - $0.622 \times 0.378 = 0.235$
   - $0.269 \times 0.731 = 0.197$
   - $0.881 \times 0.119 = 0.105$
   - $0.731 \times 0.269 = 0.197$
   - $0.378 \times 0.622 = 0.235$

   Produit : $0.235 \times 0.197 \times 0.105 \times 0.197 \times 0.235 \approx 2.2 \times 10^{-4}$ (encore 4 fois plus petit que le majorant).
3. Mise à jour ≈ $\eta \times 2.2 \times 10^{-4} = 0.1 \times 2.2 \times 10^{-4} \approx 2 \times 10^{-5}$ : la 1re couche **ne bouge quasiment pas**.
4. **Non.** $w\sigma'(0) = 1$ seulement si $z$ reste exactement à 0. Avec de grands poids, $|z|$ devient grand → la sigmoïde **sature** ($\sigma' \approx 0$) et le produit s'écroule ; et là où un facteur dépasse 1, le gradient **explose**. Équilibre instable : la vraie solution est **ReLU + bonne init**.

### ✏️ Exercice 2 · Une chaîne ReLU, forward et backward

> $z_1 = w_1 x + b_1$, $a_1 = \text{ReLU}(z_1)$ ; $z_2 = w_2 a_1 + b_2$, $a_2 = \text{ReLU}(z_2)$ ; $\hat y = w_3 a_2$ ; $L = \frac{1}{2}(\hat y - y)^2$.
> $x = 2$, $y = 3$, $w_1 = 0.5$, $b_1 = 1$, $w_2 = -1$, $b_2 = 3$, $w_3 = 2$.
> 1. Forward. 2. $\partial L/\partial w_3$, $\partial L/\partial w_2$, $\partial L/\partial w_1$. 3. Même chose avec $b_2 = 0$.
> Indice : pars de $\partial L/\partial \hat y = \hat y - y$ et multiplie par une dérivée locale à chaque étape.

**Recette** : ReLU' = 1 si $z > 0$, 0 sinon. On remonte : $\times w$ pour traverser un poids, $\times \text{ReLU}'$ pour traverser une activation, $\times$ (l'entrée) pour obtenir le gradient d'un poids.

**Correction**

1. **Forward** : $z_1 = 0.5 \times 2 + 1 = 2$, $a_1 = 2$ ; $z_2 = -1 \times 2 + 3 = 1$, $a_2 = 1$ ; $\hat y = 2 \times 1 = 2$ ; $L = \frac{1}{2}(2 - 3)^2 = 0.5$.
2. **Backward** : $\partial L/\partial \hat y = 2 - 3 = -1$.
   - $\partial L/\partial w_3 = (\hat y - y) \cdot a_2 = -1 \times 1 = $ **−1**
   - $\partial L/\partial z_2 = (\hat y - y) \cdot w_3 \cdot \text{ReLU}'(z_2) = -1 \times 2 \times 1 = -2$ → $\partial L/\partial w_2 = -2 \times a_1 = $ **−4**
   - $\partial L/\partial z_1 = \partial L/\partial z_2 \cdot w_2 \cdot \text{ReLU}'(z_1) = -2 \times (-1) \times 1 = 2$ → $\partial L/\partial w_1 = 2 \times x = $ **4**
3. **$b_2 = 0$** : $z_2 = -2 + 0 = -2 < 0$ → $a_2 = 0$, $\hat y = 0$, $L = \frac{1}{2}(0 - 3)^2 = 4.5$.
   $\text{ReLU}'(z_2) = 0$ → $\partial L/\partial w_2 = 0$ **et** $\partial L/\partial w_1 = 0$ (et aussi $\partial L/\partial w_3 = -3 \times a_2 = 0$).
   Le neurone 2 est **éteint** : **aucun gradient** ne passe, alors que la loss est grande. Si c'est vrai pour toutes les entrées, c'est un **dead ReLU**.

### ✏️ Exercice 3 · Les poids initiaux d'une couche 784-256

> 1. Écart-type des poids de `nn.Linear(784, 256)` avec Xavier $2/(n_\text{in} + n_\text{out})$, Xavier $1/n_\text{in}$, He.
> 2. Même couche avec $\mathcal{N}(0, 1)$, entrées standardisées (variance 1) : écart-type de $z$ ? Que fait la sigmoïde avec ça ?
> 3. Quelle init pour le réseau ReLU à 6 couches du TP ? Pour un réseau tanh ?
> Indice : $\sqrt{0.001923} \approx 0.0439$, $\sqrt{0.001276} \approx 0.0357$, $\sqrt{0.002551} \approx 0.0505$.

**Recette** : écart-type = $\sqrt{\text{variance}}$.

**Correction**

1. - Xavier $2/(784 + 256) = 2/1040 = 0.001923$ → std **0.0439**
   - Xavier $1/784 = 0.001276$ → std **0.0357**
   - He $2/784 = 0.002551$ → std **0.0505** (c'est la valeur affichée par `layer.weight.std()` dans le cours)
2. $\text{Var}(z) = 784 \times 1 \times 1 = 784$ → std $= \sqrt{784} = $ **28**. La sigmoïde **sature** : $\sigma(z) \approx 0$ ou 1 presque partout, $\sigma' \approx 0$ → gradient nul, rien n'apprend.
3. ReLU → **He**. tanh → **Xavier**.

### ⚠️ Pièges QCM · Séance 5

- Réseau profond sigmoïde qui reste à 10 % / loss 2.30 → **vanishing gradient**, pas un bug.
- $\max \sigma' = 1/4$ → $L$ couches divisent le gradient par au moins $4^L$.
- ReLU' = 1 sur le chemin actif ; ReLU reste **non linéaire**.
- Dead ReLU → Leaky ReLU / lr plus petit / init He.
- Init à zéro → **symétrie** (pas "vanishing").
- He = $2/n_\text{in}$ (ReLU) ; Xavier = $1/n_\text{in}$ ou $2/(n_\text{in}+n_\text{out})$ (tanh/sigmoïde).
- Standardiser avec les stats du **train** uniquement.

---

## Séance 6 · Mieux entraîner un réseau

### 🧒 L'histoire

**L'overfitting**, c'est l'élève qui **apprend les annales par cœur** : 20/20 sur les annales (train), 12/20 à l'examen (validation). Les remèdes :

- **Weight decay** : une petite **taxe sur les gros poids** → le modèle reste simple.
- **Dropout** : à chaque entraînement, des **coéquipiers absents au hasard** → chacun doit savoir tout faire, personne ne dépend d'un seul.
- **Early stopping** : **arrêter de réviser** quand les notes au bac blanc commencent à baisser, et garder la meilleure version.
- **Batch norm** : remettre les entrées de chaque couche **à la bonne échelle**.
- **Momentum** : une **bille qui prend de l'élan** dans la descente. **Adam** : une bille avec élan **et** un pas adapté à chaque direction.

### 📌 Ce qu'il faut savoir

**Le constat** : réseau 784-512-512-10 ReLU sur seulement 2 000 images (670 000 paramètres !). Train loss → 0 ; validation loss meilleure à l'epoch 8 puis **remonte**. Après 40 epochs : train 0.998, validation 0.822.

| | Train | Validation | Cause | Remède |
| --- | --- | --- | --- | --- |
| **Underfitting** (biais élevé) | erreur haute | erreur haute | modèle trop simple ou pas assez entraîné | plus gros modèle, plus long |
| **Overfitting** (variance élevée) | erreur basse | erreur **bien plus haute** | le modèle apprend le bruit | **plus de données** d'abord, puis régulariser |

**L2 / weight decay**

$$J_\text{reg}(W) = J(W) + \frac{\lambda}{2}\|W\|^2 \qquad \frac{\partial J_\text{reg}}{\partial w} = \frac{\partial J}{\partial w} + \lambda w$$

$$w \leftarrow \underbrace{(1 - \eta\lambda)}_{\text{un peu} < 1} w - \eta\frac{\partial J}{\partial w}$$

- À chaque pas, le poids est d'abord multiplié par $1 - \eta\lambda$ ("**decay**" vers 0), puis le pas habituel.
- Un poids inutile aux données tend vers 0 ; un poids utile est maintenu par son gradient.
- $\lambda$ typique : $10^{-5}$ à $10^{-2}$. Les biais ne sont généralement pas pénalisés.
- PyTorch : `SGD(..., weight_decay=1e-3)` ; pour Adam, **`AdamW`** (version découplée).

**Dropout** (Srivastava, Hinton et al., 2014)
- En **entraînement**, chaque neurone caché est mis à 0 avec probabilité $p$ (typiquement 0.1 à 0.5). Chaque pas entraîne un réseau "aminci" différent.
- En **test** : pas de dropout, le réseau complet.
- **Inverted dropout** : masque $m_i \in \{0, 1\}$ avec $P(m_i = 1) = 1 - p$ :

$$\tilde a_i = \frac{m_i\,a_i}{1 - p}\ \text{(entraînement)}, \qquad \tilde a_i = a_i\ \text{(évaluation)}, \qquad \mathbb{E}[\tilde a_i] = \frac{a_i(1-p)}{1-p} = a_i$$

- On divise par $1 - p$ pour que la couche suivante reçoive **en moyenne** le même signal en train et en test.
- `nn.Dropout(p=0.5)` : **p = probabilité d'ÉTEINDRE**. Les valeurs gardées sont multipliées par $1/(1-p) = 2$.
- **Bug classique** : oublier `model.eval()` avant d'évaluer (prédictions aléatoires) ; oublier `model.train()` après (dropout désactivé en silence).

**Early stopping** : après chaque epoch, si la val loss s'améliore → on sauvegarde (`copy.deepcopy(model.state_dict())`) et `wait = 0` ; sinon `wait += 1` ; si `wait >= patience` → stop. À la fin, on **recharge le meilleur** état.

**Batch normalisation** (Ioffe et Szegedy, 2015). Pour un neurone, sur un mini-batch de $m$ valeurs :

$$\mu_B = \frac{1}{m}\sum_i z^{(i)}, \quad \sigma_B^2 = \frac{1}{m}\sum_i (z^{(i)} - \mu_B)^2, \quad \hat z^{(i)} = \frac{z^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \quad y^{(i)} = \gamma\hat z^{(i)} + \beta$$

- $\hat z$ a moyenne 0, variance 1 sur le batch ; $\varepsilon \approx 10^{-5}$ évite la division par 0.
- $\gamma, \beta$ **appris** (une paire par neurone) : le réseau peut choisir une autre échelle, voire annuler la normalisation.
- Placement : **Linear → BatchNorm1d → ReLU**.
- Avantages : entraînement plus rapide, tolère de plus grands lr, moins sensible à l'init.
- **En test** : on peut prédire une image seule → on utilise des **statistiques courantes** (moyenne mobile, momentum 0.1) : $\mu_\text{run} \leftarrow 0.9\mu_\text{run} + 0.1\mu_B$ et $\sigma^2_\text{run} \leftarrow 0.9\sigma^2_\text{run} + 0.1\frac{m}{m-1}\sigma_B^2$. Utilisées en mode `eval()`.

**Layer norm** : statistiques d'**un exemple** sur **ses propres features**. Indépendante de la taille du batch, identique en train et en eval → standard des **transformers et RNN**. BatchNorm = standard des **CNN** (ResNet), demande des batchs raisonnables (32+).

**Momentum**

$$v \leftarrow \beta v + g, \qquad w \leftarrow w - \eta v \qquad (\beta = 0.9 \text{ typiquement, } v = 0 \text{ au départ})$$

- $v = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \dots$ : somme des gradients passés.
- Les composantes qui **changent de signe** (zigzag) s'annulent ; celles qui gardent le même signe **s'additionnent**.
- Gradient constant : $v \to g/(1-\beta) = 10g$ → pas jusqu'à **10 fois plus grands** dans une direction cohérente.
- Convention PyTorch (`SGD(..., momentum=0.9)`). Certains livres écrivent $v \leftarrow \beta v + (1-\beta)g$ (même idée, $\eta$ rescalé).

**Adam** (Kingma et Ba, 2015) = momentum + un pas par paramètre

$$m \leftarrow \beta_1 m + (1-\beta_1)g, \quad s \leftarrow \beta_2 s + (1-\beta_2)g^2, \quad \hat m = \frac{m}{1-\beta_1^t}, \quad \hat s = \frac{s}{1-\beta_2^t}, \quad w \leftarrow w - \eta\frac{\hat m}{\sqrt{\hat s} + \varepsilon}$$

- $m$ = gradient moyen (momentum), $s$ = carré moyen (la taille typique de $g$).
- Diviser par $\sqrt{\hat s}$ : petits gradients → **plus grands pas**, grands gradients → plus petits.
- **Correction du biais** : $m$ et $s$ partent de 0, donc après le 1er pas $m = 0.1g$ (trop petit) ; diviser par $1 - \beta_1^t$ corrige ($\hat m = g$ en $t = 1$). La correction disparaît quand $t$ grandit.
- Défauts : $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.
- Sur une vallée étroite : SGD zigzague puis rampe ; momentum et Adam atteignent le minimum.

**Schedules de learning rate** : grands pas au début, petits à la fin. **Step decay** (÷10 toutes les quelques epochs), **cosine** (descente douce vers 0), **warmup** (montée linéaire au début ; standard pour les transformers). `scheduler.step()` **une fois par epoch, après** les pas de l'optimiseur.

**Diagnostiquer un entraînement**

| Symptôme | Cause probable | Correctif |
| --- | --- | --- |
| Loss énorme puis **nan** | **lr trop grand**, gradients qui explosent | diviser $\eta$ par 10 ; gradient clipping ; vérifier la normalisation des entrées |
| Loss **plate dès le début à $\ln C$** | $\eta$ trop petit ; **bug** (pas de `step()`, pas de `zero_grad()`, labels mélangés) ; dead ReLUs ; réseau sigmoïde profond | sur-apprendre **un seul batch** d'abord ; vérifier la boucle ; ReLU + He, batch norm |
| Train ↓, **validation ↑** | **overfitting** | plus de données, weight decay, dropout, early stopping |
| Les deux **hautes et proches** | **underfitting** | plus gros modèle, plus long, $\eta$ plus grand, moins de régularisation |
| Loss très bruitée, ne se stabilise pas | $\eta$ trop grand en fin d'entraînement, batch trop petit | schedule (decay), plus gros batch |

> Avec $C$ classes, un réseau qui ne sait rien prédit $1/C$ partout → loss $= -\ln(1/C) = \ln C$ ($\ln 10 = 2.30$ pour FashionMNIST).

**Gradient clipping** : si $\|g\| > c$ (norme de **tout** le vecteur gradient), $g \leftarrow c\,g/\|g\|$. **La direction est gardée**, seule la longueur est limitée. Exemple : $g = (3, 4)$, $\|g\| = 5$, $c = 1$ → $g = (0.6, 0.8)$. Se place **entre `backward()` et `step()`**. Standard pour les RNN.

```python
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### ✏️ Exercice 1 · Une mise à jour avec weight decay

> $w = (2, -1)$, $\partial J/\partial w = (0.5, -0.5)$, $\eta = 0.1$.
> 1. Nouveaux poids sans régularisation. 2. Avec weight decay $\lambda = 0.5$ : facteur $1 - \eta\lambda$ ?
> 3. Pénalité $\frac{\lambda}{2}\|w\|^2$ au départ. 4. Si le gradient des données était 0, en combien de pas un poids serait-il divisé par 2 ?

**Correction**

1. $w - \eta g = (2 - 0.05,\ -1 + 0.05) = (1.95,\ -0.95)$.
2. Facteur $1 - 0.1 \times 0.5 = $ **0.95**. $w = 0.95 \times (2, -1) - 0.1 \times (0.5, -0.5) = (1.9 - 0.05,\ -0.95 + 0.05) = (1.85,\ -0.90)$.
3. $\frac{0.5}{2}(2^2 + (-1)^2) = 0.25 \times 5 = $ **1.25**.
4. On veut $0.95^n = 0.5$ → $n = \ln 0.5 / \ln 0.95 = 13.5$ → **environ 14 pas**.

### ✏️ Exercice 2 · Appliquer un masque de dropout

> $a = (2, 1, 4, 3)$, $p = 0.5$, masque $m = (1, 0, 1, 0)$.
> 1. $\tilde a$ en entraînement (inverted dropout). 2. Compare les sommes. Problème ?
> 3. Espérance de $\tilde a_1$ sur tous les masques. 4. Sortie en mode évaluation ?

**Correction**

1. $\tilde a = m \cdot a / (1 - 0.5) = (2, 0, 4, 0) \times 2 = (4,\ 0,\ 8,\ 0)$.
2. Somme de $\tilde a$ = 12, somme de $a$ = 10. **Pas un problème** : un masque particulier donne un écart, mais **en moyenne** sur les masques c'est égal ; c'est le but du facteur $1/(1-p)$.
3. $\mathbb{E}[\tilde a_1] = 0.5 \times 4 + 0.5 \times 0 = 2 = a_1$ ✅
4. En évaluation : identité → $(2,\ 1,\ 4,\ 3)$.

### ✏️ Exercice 3 · Batch norm sur un batch de 4 nombres

> $z = (1, 3, 3, 9)$, $\gamma = 2$, $\beta = 0.5$, on néglige $\varepsilon$.
> 1. $\mu_B$ et $\sigma_B^2$. 2. $\hat z$, vérifier la moyenne 0. 3. $y = \gamma\hat z + \beta$. 4. Running mean (départ 0, momentum 0.1) après ce batch ?

**Correction**

1. $\mu_B = (1 + 3 + 3 + 9)/4 = 16/4 = $ **4**.
   $\sigma_B^2 = [(1-4)^2 + (3-4)^2 + (3-4)^2 + (9-4)^2]/4 = (9 + 1 + 1 + 25)/4 = 36/4 = $ **9** (donc $\sigma_B = 3$).
2. $\hat z = (z - 4)/3 = (-1,\ -1/3,\ -1/3,\ 5/3) \approx (-1,\ -0.333,\ -0.333,\ 1.667)$. Somme : $-1 - 0.667 + 1.667 = 0$ ✅
3. $y = 2\hat z + 0.5 = (-1.5,\ -0.167,\ -0.167,\ 3.833)$.
4. $\mu_\text{run} = 0.9 \times 0 + 0.1 \times 4 = $ **0.4**.

### ✏️ Exercice 4 · Deux pas de SGD, momentum et Adam

> $J(w) = w^2$ (donc $g = 2w$), départ $w = 1$, $\eta = 0.1$.
> 1. Descente de gradient simple : $w$ après 1 et 2 pas.
> 2. Momentum $\beta = 0.9$, $v = 0$ : $g, v, w$ après 1 et 2 pas. Qui est le plus près du minimum ?
> 3. Adam, premier pas ($m = s = 0$, on néglige $\varepsilon$) : montrer $\hat m = g$, $\hat s = g^2$. Taille du premier pas ?

**Correction**

1. **SGD** : pas 1 : $g = 2$, $w = 1 - 0.1 \times 2 = 0.8$. Pas 2 : $g = 1.6$, $w = 0.8 - 0.16 = $ **0.64**.
2. **Momentum** :

| Pas | $g = 2w$ | $v = 0.9v + g$ | $w = w - 0.1v$ |
| --- | --- | --- | --- |
| 1 | 2 | $0 + 2 = 2$ | $1 - 0.2 = 0.8$ |
| 2 | 1.6 | $0.9 \times 2 + 1.6 = 3.4$ | $0.8 - 0.34 = $ **0.46** |

   Le momentum est **plus proche du minimum** (0.46 < 0.64) : l'élan accélère dans une direction cohérente.
3. **Adam** pas 1 : $m = 0.1g$, $s = 0.001g^2$. $\hat m = 0.1g / (1 - 0.9) = g$ ; $\hat s = 0.001g^2/(1 - 0.999) = g^2$.
   Pas $= \eta\,\hat m/\sqrt{\hat s} = \eta\,g/|g| = \eta \times \text{signe}(g)$ → taille **$\eta = 0.1$, quel que soit $g$**. Ici $w = 1 - 0.1 = 0.9$.

### ✏️ Exercice 5 · Quatre runs à diagnostiquer

> - **Run A** : train loss descend vers ~0.05 ; validation descend jusqu'à ~0.7 vers l'epoch 15, puis **remonte** vers 1.0.
> - **Run B** : la loss **explose** (échelle log, jusqu'à 100+) puis **nan à partir de l'epoch 8**.
> - **Run C** : train et validation **plates à 2.30** pendant 40 epochs.
> - **Run D** : les deux descendent de 2.0 à ~**1.25** et restent **proches et hautes**.
>
> Que se passe-t-il, que changes-tu en premier ? Pourquoi C est-il à 2.30 exactement ?

**Correction**

| Run | Diagnostic | Premier changement |
| --- | --- | --- |
| **A** | **Overfitting** : l'écart train/val grandit | **early stopping** (garder l'epoch ~15), puis weight decay / dropout, plus de données |
| **B** | **lr trop grand** → divergence, gradients qui explosent → nan | **diviser le lr par 10** ; gradient clipping ; vérifier la normalisation des entrées |
| **C** | **Rien n'est appris** : le réseau prédit 1/10 pour chaque classe | **sur-apprendre un seul batch** pour trouver le bug (step, zero_grad, labels) ; sinon lr, dead ReLUs, sigmoïdes profondes → ReLU + He |
| **D** | **Underfitting** : train et val hautes et proches | **plus gros modèle**, entraîner plus longtemps, lr plus grand, moins de régularisation |

**Pourquoi 2.30** : 10 classes, prédiction uniforme $p = 1/10$ → loss $= -\ln(1/10) = \ln 10 = 2.30$.

### ⚠️ Pièges QCM · Séance 6

- Dans `nn.Dropout(p)`, $p$ = proba de **supprimer** (pas de garder). Facteur $1/(1-p)$ en **train**, rien en eval.
- Weight decay = multiplier par $(1 - \eta\lambda)$ avant le pas.
- BatchNorm : stats **du batch** en train, **running stats** en eval. LayerNorm : identique en train et eval.
- Le premier pas d'Adam a une taille ≈ $\eta$.
- Momentum $\beta = 0.9$ → jusqu'à 10× plus grand.
- Gradient clipping garde la **direction**, se place entre `backward()` et `step()`.
- Loss plate à $\ln C$ → rien n'est appris ; nan → lr trop grand.

---

## Séance 7 · Les réseaux convolutifs (CNN)

### 🧒 L'histoire

Avec une couche dense, on **aplatit** l'image et on mélange tous les pixels : le réseau ne sait même pas quels pixels sont **voisins**. Un CNN fait autrement : il **promène une petite loupe** (le **filtre**, 3×3) sur toute l'image. La loupe cherche un **motif** (un bord, une texture). **La même loupe partout** = partage des poids : si une botte apparaît ailleurs, on la reconnaît sans réapprendre. Ensuite on **résume** (pooling) et on **empile** : bords → textures → parties d'objets → objets.

### 📌 Ce qu'il faut savoir

**Les 3 problèmes du dense sur les images**
1. **Trop de paramètres** (chaque neurone relié à chaque pixel ; une photo 224×224×3 → explosion).
2. **Pas de voisinage** (mélanger les pixels de la même façon dans toutes les images ne change rien pour un dense).
3. **Pas de translation** (un objet au centre doit être réappris pour chaque position).

**La convolution**

$$Y_{i,j} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K_{u,v}\,X_{i+u,\,j+v} + b$$

- On pose le filtre $k \times k$ sur un patch, on multiplie élément par élément, on **somme** (+ biais), on décale d'un pixel, on recommence.
- Sans padding, une image $n \times n$ donne une sortie $(n - k + 1) \times (n - k + 1)$ : la **feature map**.
- En vrai c'est une **cross-corrélation** (une vraie convolution retourne le noyau) ; comme $K$ est appris, ça ne change rien.
- Un filtre = un **détecteur de motif** : la sortie est grande là où le patch ressemble au noyau. Les noyaux sont **appris** par backprop. Un noyau 3×3 = 9 poids utilisés **à toutes les positions** = **weight sharing**.

**Padding et stride**

$$n_\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

| $n$ | $k$ | $p$ | $s$ | $n_\text{out}$ |
| --- | --- | --- | --- | --- |
| 5 | 3 | 0 | 1 | 3 |
| 28 | 3 | 1 | 1 | 28 ("same") |
| 28 | 2 | 0 | 2 | 14 |
| 224 | 7 | 3 | 2 | 112 |

- **Padding "same"** pour un noyau impair : $p = (k-1)/2$ → 1 pour 3×3, 3 pour 7×7.
- **Stride 2** divise la taille par 2.

**Canaux**
- Un filtre a la shape $C_\text{in} \times k \times k$ : il couvre **tous** les canaux d'entrée et produit **une** feature map. $C_\text{out}$ filtres → $C_\text{out}$ canaux en sortie.

$$\#\text{params} = C_\text{out} \times (C_\text{in} \times k \times k + 1) \qquad \text{indépendant de la taille de l'image } n$$

- PyTorch : images **(N, C, H, W)** = (batch, canaux, hauteur, largeur). `nn.Conv2d(3, 16, 3, padding=1).weight.shape` = **(16, 3, 3, 3)**, `bias.shape` = (16,).

**Pooling**
- **Max pooling** 2×2 stride 2 : garde le max de chaque fenêtre 2×2. **Average pooling** : la moyenne.
- Divise H et W par 2 (4 fois moins de valeurs). **Aucun paramètre**. Un peu d'**invariance par translation**.
- Bloc typique : conv, ReLU, conv, ReLU, pool. Le nombre de canaux **double** souvent quand la taille est divisée par 2.

**Champ récepteur** (la zone de l'entrée dont dépend un neurone) : on part de $r = 1$, $j = 1$ et pour chaque couche :

$$r \leftarrow r + (k - 1)\,j, \qquad j \leftarrow j \times s$$

Les strides et le pooling multiplient $j$ : après eux, chaque nouvelle couche agrandit $r$ plus vite. Premières couches : bords ; couches profondes : parties d'objets, puis objets.

**Biais inductif** : ce que l'architecture suppose avant de voir les données : **localité** et **partage des poids**. Bonnes hypothèses → moins de paramètres et meilleure généralisation. CNN du cours : **20 490** paramètres contre 235 146 pour le dense de la séance 5.

```python
cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 14 x 14
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 7 x 7
    nn.Flatten(), nn.Linear(32 * 7 * 7, 10))                      # 20 490 paramètres
```
(conv1 : $16 \times 10 = 160$ ; conv2 : $32 \times 145 = 4640$ ; linear : $1568 \times 10 + 10 = 15\,690$.)

**Historique**

| Réseau | Année | Idée | Paramètres |
| --- | --- | --- | --- |
| LeNet-5 (LeCun) | 1998 | conv, pool, conv, pool, dense ; lit les chiffres des chèques | ~60 000 |
| AlexNet (Krizhevsky, Sutskever, Hinton) | 2012 | même recette en plus gros, **ReLU, dropout, GPU** ; gagne ImageNet | ~60 millions |
| VGG-16 | 2014 | **que des 3×3** empilés, 16 couches | ~138 millions |
| ResNet (He et al.) | 2015 | **connexions résiduelles**, 50 / 101 / 152 couches | ResNet-18 : 11.7 M |

**Plus profond n'est pas forcément mieux : le problème de dégradation.** Sur CIFAR-10, un CNN "plain" de 56 couches a une erreur **d'entraînement** plus haute qu'un de 20 couches. Ce n'est **pas de l'overfitting** (c'est le train qui est mauvais) : c'est un problème **d'optimisation**.

**ResNet** : chaque bloc apprend une **correction** $F(x)$ ajoutée à son entrée :

$$y = F(x) + x \qquad \frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$$

Un bloc inutile n'a qu'à apprendre $F = 0$. Le gradient qui arrive au bloc $l$ contient toujours $\partial J/\partial x_L$ **lui-même** (le terme $I$), pas multiplié par des poids ou des dérivées d'activation → il **ne peut pas disparaître** par un produit de petits facteurs. (15 blocs : gradient au bloc 1 de $4 \times 10^{-14}$ en plain contre $6 \times 10^{-2}$ en résiduel.)

**Transfer learning** : un ResNet entraîné sur ImageNet a appris des features générales (bords et couleurs au début, textures et parties au milieu, features spécifiques à la fin). Pour une nouvelle tâche avec peu d'images : on **garde le réseau**, on **remplace la dernière couche** (1000 classes ImageNet) par une nouvelle. On prépare les entrées comme au pré-entraînement : resize, normalisation ImageNet (moyenne $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$).

| Tes données | Stratégie |
| --- | --- |
| Peu d'images, proches de photos | **tout geler**, entraîner la nouvelle tête |
| Peu à moyen, un peu différentes | puis **dégeler le dernier bloc**, petit lr |
| Beaucoup d'images ou très différentes (radios, peintures) | **fine-tuner tout**, petit lr |

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)   # 11 689 512 paramètres
for p in model.parameters():
    p.requires_grad = False                          # geler le backbone
model.fc = nn.Linear(512, 10)                        # nouvelle tête : 5 130 paramètres
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
# puis fine-tuning du dernier bloc avec un lr plus petit (groupes de paramètres) :
for p in model.layer4.parameters():
    p.requires_grad = True
optimizer = torch.optim.Adam([{"params": model.layer4.parameters(), "lr": 1e-4},
                              {"params": model.fc.parameters(), "lr": 1e-3}])
```

`requires_grad = False` : autograd ne calcule pas de gradient pour ces poids, l'optimiseur ne les bouge pas.

### ✏️ Exercice 1 · Une couche dense sur une photo

> Photo couleur 224×224×3 aplatie, couche dense de 1000 neurones.
> 1. Nombre d'entrées ? 2. Nombre de paramètres ? 3. Mémoire en float32 (4 octets) ?
> 4. Une conv de 64 filtres 3×3 sur 3 canaux a $64 \times (3 \times 3 \times 3 + 1)$ paramètres : combien ? Dépend-il de la taille de l'image ?

**Correction**

1. $224 \times 224 \times 3 = $ **150 528** entrées.
2. $150\,528 \times 1000 + 1000 = $ **150 529 000** (~150 millions).
3. $150\,529\,000 \times 4 \approx 602$ millions d'octets ≈ **602 Mo** (~0.6 Go), pour **une seule** couche.
4. $64 \times 28 = $ **1 792** paramètres, **indépendant** de la taille de l'image (le même filtre glisse partout).

### ✏️ Exercice 2 · Une convolution à la main

> $X = \begin{pmatrix}1&1&1&4&4\\1&2&1&4&4\\1&1&1&4&4\\1&1&2&4&4\\1&1&1&4&4\end{pmatrix}$, $\quad K = \begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$
>
> 1. Taille de la sortie (pas de padding, stride 1) ? 2. Montrer que chaque sortie = (somme colonne gauche du patch) − (somme colonne droite).
> 3. Calculer $Y$. 4. Où $|Y|$ est-il grand ? Que détecte ce noyau ? Que donnerait $-K$ ?

**Recette** : pose le 3×3 en haut à gauche, calcule, décale d'une case vers la droite ; en fin de ligne, redescends d'une case.

**Correction**

1. $5 - 3 + 1 = 3$ → sortie **3×3**.
2. Le noyau a une colonne de 1, une colonne de 0, une colonne de −1 : $\sum K \cdot X = 1 \times (\text{colonne gauche}) + 0 \times (\text{milieu}) - 1 \times (\text{colonne droite})$.
3. Les colonnes de $X$ : $c_0 = (1,1,1,1,1)$, $c_1 = (1,2,1,1,1)$, $c_2 = (1,1,1,2,1)$, $c_3 = c_4 = (4,4,4,4,4)$.
   - Colonne de sortie 0 (gauche $c_0$, droite $c_2$) : $3 - 3 = 0$ ; $3 - 4 = -1$ ; $3 - 4 = -1$
   - Colonne de sortie 1 (gauche $c_1$, droite $c_3$) : $4 - 12 = -8$ ; $4 - 12 = -8$ ; $3 - 12 = -9$
   - Colonne de sortie 2 (gauche $c_2$, droite $c_4$) : $3 - 12 = -9$ ; $4 - 12 = -8$ ; $4 - 12 = -8$

$$Y = \begin{pmatrix}0 & -8 & -9\\ -1 & -8 & -8\\ -1 & -9 & -8\end{pmatrix}$$

4. $|Y|$ est grand dans les **colonnes 2 et 3 de la sortie**, là où le patch chevauche le **saut de 1 à 4**. Ce noyau détecte les **bords verticaux** (changement d'intensité de gauche à droite). Négatif car l'image devient **plus claire** vers la droite. Les petits −1 viennent des "2" isolés (du bruit). **$-K$** donne les **mêmes valeurs avec le signe opposé** (+8, +9…) : même bord, polarité inversée.

### ✏️ Exercice 3 · Shapes et paramètres le long d'un réseau

> Entrée : une image grise 1×28×28. Shape $(C, H, W)$ et nombre de paramètres après chaque couche :
> 1. `Conv2d(1, 8, 3, padding=1)` 2. `MaxPool2d(2)` 3. `Conv2d(8, 16, 3)` 4. `MaxPool2d(2)` 5. `Conv2d(16, 32, 3, stride=2, padding=1)` 6. `Flatten()`

**Recette** : $C$ = nombre de filtres ; $H = W = \lfloor(n + 2p - k)/s\rfloor + 1$ ; params $= C_\text{out}(C_\text{in}k^2 + 1)$ ; pooling = taille ÷ 2, 0 paramètre.

**Correction**

| Couche | Calcul de la taille | Shape | Paramètres |
| --- | --- | --- | --- |
| `Conv2d(1, 8, 3, padding=1)` | $(28 + 2 - 3)/1 + 1 = 28$ | (8, 28, 28) | $8 \times (1 \times 9 + 1) = $ **80** |
| `MaxPool2d(2)` | $28/2$ | (8, 14, 14) | 0 |
| `Conv2d(8, 16, 3)` | $14 - 3 + 1 = 12$ | (16, 12, 12) | $16 \times (8 \times 9 + 1) = $ **1 168** |
| `MaxPool2d(2)` | $12/2$ | (16, 6, 6) | 0 |
| `Conv2d(16, 32, 3, stride=2, padding=1)` | $\lfloor(6 + 2 - 3)/2\rfloor + 1 = 2 + 1 = 3$ | (32, 3, 3) | $32 \times (16 \times 9 + 1) = $ **4 640** |
| `Flatten()` | $32 \times 3 \times 3$ | **288** nombres | 0 |

Total : 5 888 paramètres.

### ✏️ Exercice 4 · Pooling à la main

> $X = \begin{pmatrix}1&3&2&0\\4&6&1&1\\0&2&5&7\\1&1&3&2\end{pmatrix}$
> 1. Max pooling 2×2 (stride 2). 2. Average pooling 2×2. 3. Shape de `nn.MaxPool2d(2)` sur un tenseur 64×32×32 ? Paramètres ?

**Recette** : découpe en 4 carrés 2×2 qui ne se chevauchent pas.

**Correction**

| Fenêtre | Valeurs | Max | Moyenne |
| --- | --- | --- | --- |
| haut-gauche | 1, 3, 4, 6 | 6 | 14/4 = 3.5 |
| haut-droite | 2, 0, 1, 1 | 2 | 4/4 = 1 |
| bas-gauche | 0, 2, 1, 1 | 2 | 4/4 = 1 |
| bas-droite | 5, 7, 3, 2 | 7 | 17/4 = 4.25 |

1. Max : $\begin{pmatrix}6 & 2\\ 2 & 7\end{pmatrix}$ 2. Moyenne : $\begin{pmatrix}3.5 & 1\\ 1 & 4.25\end{pmatrix}$
3. **64×16×16** (les canaux ne changent pas, H et W divisés par 2), **0 paramètre**.

### ✏️ Exercice 5 · Champs récepteurs

> 1. Trois convs 3×3 empilées (stride 1) : champ récepteur ?
> 2. Avec $C$ canaux en entrée et sortie partout, poids de ces trois 3×3 contre une seule 7×7 (sans biais) ? Laquelle a plus de non-linéarités ?
> 3. conv 3×3 (stride 1), max pool 2×2 (stride 2), conv 3×3 (stride 1) : champ récepteur ?

**Recette** : $r = 1$, $j = 1$ ; pour chaque couche $r \leftarrow r + (k-1)j$ puis $j \leftarrow j \times s$.

**Correction**

1. $r = 1 \to 3 \to 5 \to 7$ → **7×7**.
2. Trois 3×3 : $3 \times (C \times C \times 9) = 27C^2$. Une 7×7 : $49C^2$. Les trois 3×3 ont **moins de poids** (27 contre 49) **et plus de non-linéarités** (3 ReLU contre 1), pour le **même** champ récepteur. C'est l'idée de VGG.
3. - conv 3×3 : $r = 1 + 2 \times 1 = 3$, $j = 1$
   - pool 2×2 stride 2 : $r = 3 + 1 \times 1 = 4$, $j = 2$
   - conv 3×3 : $r = 4 + 2 \times 2 = $ **8**

### ⚠️ Pièges QCM · Séance 7

- Paramètres d'une conv : **$C_\text{out}(C_\text{in}k^2 + 1)$**, ne dépend **pas** de la taille de l'image. N'oublie pas $C_\text{in}$ ni le biais.
- Pooling : **0 paramètre**, ne change pas le nombre de canaux.
- $n_\text{out} = \lfloor(n + 2p - k)/s\rfloor + 1$ : arrondi **vers le bas**.
- PyTorch : **(N, C, H, W)** ; poids d'une conv : **(C_out, C_in, k, k)**.
- 56 couches pire que 20 en **train** → problème d'**optimisation** (dégradation), pas d'overfitting.
- ResNet : le terme **identité** $+I$ laisse passer le gradient.
- Transfer learning peu de données → **geler** et entraîner la tête ; petit lr pour le fine-tuning.

---

## Séance 8 · RNN, LSTM, attention

### 🧒 L'histoire

**Le RNN** lit une phrase **mot par mot** en gardant un **résumé dans la tête** (l'état caché $h$). À chaque mot, il met à jour son résumé avec **les mêmes règles** (les mêmes poids). Problème : le résumé **s'efface vite**, on oublie le début d'une longue phrase. C'est le vanishing gradient, mais dans le **temps**.

**Le LSTM** ajoute un **carnet de notes** (la *cell state*) avec trois **portes** : « j'efface ? » (*forget*), « j'écris ? » (*input*), « je le dis ? » (*output*). Comme on **ajoute** au carnet au lieu de tout réécrire, l'information traverse beaucoup d'étapes.

**L'attention** : au lieu de tout compresser dans un seul résumé, à chaque mot qu'on écrit, on **relit toute la phrase d'entrée** en regardant **surtout les passages utiles** (des poids qui somment à 1).

### 📌 Ce qu'il faut savoir

**Séquences** : texte, séries temporelles, audio, ADN… Deux propriétés : **l'ordre compte**, la **longueur varie**. Un dense veut une taille fixe ; un CNN ne voit qu'une fenêtre ; aucun n'a de mémoire. Séance 7 : partage des poids dans l'**espace** ; séance 8 : partage des poids dans le **temps**.

**Le neurone récurrent**

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b), \qquad h_0 = 0$$

- $h_t$ ($n_h$ nombres) résume tout ce qui a été vu jusqu'à $t$. $W_{xh}$ : $(n_h, n_\text{in})$, $W_{hh}$ : $(n_h, n_h)$.
- **Mêmes poids à chaque pas** → le nombre de paramètres **ne dépend pas de la longueur**.
- Sortie : $\hat y = W_{hy}h_T$ (un label par séquence, *many-to-one*) ou $\hat y_t = W_{hy}h_t$ (un par pas, *many-to-many*).

**En PyTorch**

```python
rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(32, 20, 8)        # 32 séquences, 20 pas, 8 features
out, h_n = rnn(x)
out.shape    # (32, 20, 16)  tous les états h_1 ... h_20
h_n.shape    # (1, 32, 16)   le dernier (1 couche)
# paramètres : 16*8 + 16*16 + 2*16 = 416 (deux vecteurs de biais)
```

- `batch_first=True` : entrées **(batch, temps, features)**.
- Classifier une séquence : couche linéaire sur `out[:, -1]` (le dernier état). Prédire à chaque pas : sur chaque `out[:, t]`.

**Backpropagation through time (BPTT)** : déroulé, un RNN = un réseau profond **avec une couche par pas de temps** et des poids partagés :

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T}\frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T}\text{diag}(\tanh'(z_t))\,W_{hh}$$

Cas scalaire : produit de $T - 1$ facteurs $w_h\tanh'(z_t)$ avec $0 < \tanh' \le 1$.
- $|w_h| < 1$ (ou tanh saturée) → **vanishing** : le passé lointain ne reçoit aucun gradient, pas de dépendances longues.
- $|w_h| > 1$ → **exploding** : pas énormes, nan → **gradient clipping**.
- Même maladie que les réseaux sigmoïdes profonds (séance 5), mais la profondeur = **la longueur de la séquence**.

**LSTM** (Hochreiter et Schmidhuber, 1997). Avec $u_t = [h_{t-1}, x_t]$ :

| Porte | Formule | Rôle |
| --- | --- | --- |
| forget $f_t$ | $\sigma(W_f u_t + b_f)$ | que garder de l'ancienne mémoire |
| input $i_t$ | $\sigma(W_i u_t + b_i)$ | combien écrire |
| candidat $g_t$ | $\tanh(W_g u_t + b_g)$ | quoi écrire |
| output $o_t$ | $\sigma(W_o u_t + b_o)$ | combien montrer |
| cell state | $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ | la mémoire (**additive**) |
| hidden | $h_t = o_t \odot \tanh(c_t)$ | la sortie |

- Les portes sont des **sigmoïdes** (entre 0 et 1 : laisser passer ou non).
- **Pourquoi il se souvient** : le long du cell state, $\dfrac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$ (chemin direct) : **pas de matrice de poids, pas de tanh'** sur ce chemin, seulement la porte forget. Si $f_t \approx 1$, information et gradient traversent beaucoup d'étapes. **Même idée que ResNet** ($y = x + F(x)$) : un chemin identité pour le gradient.
- **Initialiser le biais de forget à 1 ou 2** pour que $f_t$ démarre proche de 1.

**GRU** (Cho et al., 2014) : un LSTM allégé, **pas de cell state séparé, 2 portes** (update $z_t$, reset $r_t$) ; $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t$ (même astuce additive). **3 blocs** de poids au lieu de 4 → ~25 % de paramètres en moins, résultats souvent similaires.

| Cellule | Blocs de poids | Paramètres (8 → 16) |
| --- | --- | --- |
| RNN | 1 | 416 |
| GRU | 3 | 1 248 |
| LSTM | 4 (portes $i, f, g, o$ dans cet ordre) | 1 664 = 4 × 416 |

`nn.LSTM` renvoie **`out, (h_n, c_n)`** (il renvoie aussi le cell state).

**Encoder-decoder** (Sutskever et al., 2014) : un RNN encodeur lit l'entrée, son dernier état $h_T$ est donné à un RNN décodeur qui écrit la sortie mot par mot. **Problème** : toute la phrase doit tenir dans **un seul vecteur** de taille fixe → la qualité chute pour les phrases longues.

**Attention douce** (Bahdanau, Cho, Bengio, 2015). États encodeur $h_1, \dots, h_T$, état décodeur courant $s$ :

$$\text{1. scores } e_i = s^\top h_i \qquad \text{2. poids } \alpha_i = \frac{e^{e_i}}{\sum_j e^{e_j}} \qquad \text{3. contexte } c = \sum_i \alpha_i h_i$$

- Score élevé = « $h_i$ est pertinent pour ce que je produis maintenant ». Poids **positifs** qui **somment à 1**.
- Un **nouveau** contexte $c$ à chaque pas de sortie. Tout est dérivable → entraîné par backprop.
- Le score peut aussi être un petit réseau (Bahdanau) ou $s^\top W h_i$ (Luong).
- Bonus : les poids $\alpha$ montrent **où le modèle regarde** (exemple des dates réécrites en AAAA-MM-JJ : l'alignement est appris tout seul).

**Vers les transformers** (Vaswani et al., 2017, « Attention is all you need ») : on supprime le RNN, chaque position regarde toutes les positions de la même séquence (**self-attention**) :

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V, \qquad Q = XW_Q,\ K = XW_K,\ V = XW_V$$

L'exercice 3 est le cas d'**une** requête avec $K = V = H$.

| | RNN, LSTM | Transformer |
| --- | --- | --- |
| Hypothèse | ordre et récence : l'info voyage pas à pas | aucune sur l'ordre (positions ajoutées en entrée) ; toutes les positions directement reliées |
| Dépendances longues | difficiles ($T$ pas de gradient) | un seul pas |
| Calcul | **séquentiel** dans le temps | **parallèle** (GPU) |
| Données | modérées | beaucoup (d'où le pré-entraînement) |

**Sentence embeddings (outil du projet)** : un transformer pré-entraîné (`all-MiniLM-L6-v2`, 6 couches, 22.7 M paramètres) transforme une phrase en **vecteur de 384 nombres**. Sens proches → vecteurs proches, même sans mots communs. **Similarité cosinus** :

$$\cos(u, v) = \frac{u^\top v}{\|u\|\,\|v\|} \in [-1, 1]$$

Exemple : chat/chaton 0.57, chat/bourse 0.09. Projet : encoder les questions des visiteurs et les descriptions des œuvres, retrouver l'œuvre de plus grande similarité ; comparer à une baseline classique (TF-IDF).

**Checklist de production**
1. **Splits propres** décidés avant tout. **Pas de fuite** : stats de normalisation sur le train seulement ; pas de (quasi-)doublons entre splits (même artiste, patient ou jour des deux côtés). Test utilisé **une fois**.
2. **Seeds fixées** (torch, NumPy, random). Mais une seed = un tirage : rapporter **moyenne et dispersion sur 3 seeds** avant d'annoncer un petit gain.
3. **Baseline d'abord** (régression logistique, TF-IDF, tête pré-entraînée seule). Sanity checks : loss initiale ≈ $\ln C$, le réseau sait sur-apprendre **un batch**.
4. **Un changement à la fois**, chaque run loggé.
- **Learning rate d'abord** (l'hyperparamètre le plus important), warmup puis cosine ou step decay.
- **Checkpoint** du meilleur modèle en validation, early stopping.
- **Mixed precision** : forward/backward en 16 bits sur GPU ; ~2× plus rapide, moitié de mémoire d'activations, même précision (copie float32 des poids ; avec float16, la loss est **scalée** pour éviter l'underflow).
- **Gradient accumulation** : le GPU ne tient que 16 images ? Additionner les gradients de 4 batchs puis un pas = même gradient qu'un batch de 64 (en divisant la loss par 4).

**Folklore contre preuves**

| On entend | Ce que disent les preuves |
| --- | --- |
| « Adam est toujours meilleur que SGD » | Adam est plus rapide et moins sensible au lr ; un SGD + momentum bien réglé peut l'égaler ou le battre (classique en vision). Comparer chacun avec **son** meilleur lr. |
| « Un plus gros batch c'est mieux » | Plus rapide par epoch, pas meilleur en soi : adapter le lr (warmup), la généralisation peut souffrir. |
| « Entraîne plus longtemps » | Seulement tant que la **validation** s'améliore. |
| « Mon astuce donne +0.3 % » | Sur une seed, souvent du **bruit**. Répéter sur plusieurs seeds. |

### ✏️ Exercice 1 · Trois pas d'un RNN à la main

> $h_t = \tanh(w_x x_t + w_h h_{t-1} + b)$, $w_x = 1$, $w_h = 0.5$, $b = 0$, $h_0 = 0$, séquence $x = (1, 0, -1)$.
> 1. $h_1, h_2, h_3$ (4 décimales). 2. $x_2 = 0$ mais $h_2 \neq 0$. Pourquoi ?
> 3. Avec $\tanh'(z) = 1 - \tanh^2(z)$, calcule $\partial h_3/\partial x_3$ et $\partial h_3/\partial x_1$. Quelle entrée influence le plus $h_3$ ?

**Recette** : $z_t = x_t + 0.5\,h_{t-1}$, $h_t = \tanh(z_t)$. Pour la dérivée, on remonte le temps : chaque pas en arrière multiplie par $\tanh'(z_t) \times w_h$.

**Correction**

1. - $h_1 = \tanh(1 + 0) = \tanh(1) = $ **0.7616**
   - $h_2 = \tanh(0 + 0.5 \times 0.7616) = \tanh(0.3808) = $ **0.3634**
   - $h_3 = \tanh(-1 + 0.5 \times 0.3634) = \tanh(-0.8183) = $ **−0.6741**
2. $h_2$ porte la **mémoire** de $h_1$ (le terme $w_h h_1$) : l'état caché résume le passé.
3. $\tanh'$ à chaque pas : $1 - h_1^2 = 0.4200$ ; $1 - h_2^2 = 0.8679$ ; $1 - h_3^2 = 0.5455$.
   - $\partial h_3/\partial x_3 = \tanh'(z_3) \times w_x = 0.5455 \times 1 = $ **0.546**
   - $\partial h_3/\partial x_1 = \tanh'(z_3)\,w_h \times \tanh'(z_2)\,w_h \times \tanh'(z_1)\,w_x = 0.5455 \times 0.5 \times 0.8679 \times 0.5 \times 0.4200 = $ **0.050**

   **$x_3$** influence ~11 fois plus que $x_1$ : l'influence du passé **s'efface** à chaque pas (c'est le vanishing gradient dans le temps).

### ✏️ Exercice 2 · Vingt pas en arrière

> 1. En ignorant $\tanh'$, le gradient qui remonte 20 pas est multiplié par $w_h^{20}$. Calcule $0.5^{20}$ et $1.5^{20}$. Conséquences ?
> 2. Remède pour le cas qui explose ? (Séance 6.)
> 3. LSTM : le gradient le long du cell state est multiplié par la porte forget $f_t$ à chaque pas. Sur 30 pas, compare $f = 0.5$ ($\sigma(0)$, init par défaut, biais 0) et $f = 0.88$ ($\sigma(2)$).

**Correction**

1. $0.5^{20} \approx 9.5 \times 10^{-7}$ (~$10^{-6}$) → **vanishing** : le passé lointain n'apprend rien. $1.5^{20} \approx 3\,325$ → **exploding** : pas énormes, nan.
2. **Gradient clipping** (limiter la norme du gradient entre `backward()` et `step()`).
3. $0.5^{30} \approx 9.3 \times 10^{-10}$ (le gradient meurt) contre $0.88^{30} \approx 0.022$ (~2 % survit). D'où l'astuce : **initialiser le biais de forget à 1 ou 2**.

### ✏️ Exercice 3 · L'attention à la main

> $h_1 = (1, 0)$, $h_2 = (0, 1)$, $h_3 = (1, 1)$, état décodeur $s = (2, 1)$.
> 1. Scores $e_i = s^\top h_i$. 2. Poids $\alpha = \text{softmax}(e)$ (3 décimales). 3. Contexte $c$.
> 4. Quel état le décodeur « regarde »-t-il ? Que deviennent les poids si on multiplie les scores par 10 ?

**Recette** : produit scalaire → softmax (c'est l'exercice 1 de la séance 4 !) → moyenne pondérée des vecteurs.

**Correction**

1. $e_1 = 2 \times 1 + 1 \times 0 = 2$ ; $e_2 = 2 \times 0 + 1 \times 1 = 1$ ; $e_3 = 2 \times 1 + 1 \times 1 = 3$ → $e = (2, 1, 3)$.
2. $e^2 = 7.389$, $e^1 = 2.718$, $e^3 = 20.086$, somme 30.193 → $\alpha = $ **(0.245, 0.090, 0.665)** (mêmes nombres que la softmax de $(2,1,0)$, dans un autre ordre, car seules les différences comptent).
3. $c = 0.245 \times (1, 0) + 0.090 \times (0, 1) + 0.665 \times (1, 1) = (0.245 + 0.665,\ 0.090 + 0.665) = $ **(0.910, 0.755)**.
4. Surtout **$h_3$** (66.5 %). Scores × 10 → $(20, 10, 30)$ → $\alpha \approx (0.00005,\ 0.000000002,\ 0.99995)$ : l'attention devient **presque one-hot** (beaucoup plus "tranchée"). C'est pour ça qu'on divise par $\sqrt d$ dans les transformers : éviter une softmax trop saturée.

### ✏️ Question de réflexion · Le camarade et ses 87.4 %

> Un camarade annonce 87.4 % avec une nouvelle augmentation contre 87.1 % sans, **un run chacun**, mesuré **sur le test** après avoir essayé **10 variantes**. Que lui demandes-tu ?

**Réponse** :
- **Plusieurs seeds** (au moins 3) avec moyenne et dispersion : 0.3 % sur un seul run, c'est souvent du bruit.
- Il a **choisi sur le test** parmi 10 variantes → le test n'est plus une estimation honnête. Il fallait choisir sur la **validation** et utiliser le test **une seule fois**.
- **Un seul changement** à la fois, mêmes conditions (même lr réglé, même nombre d'epochs).

### ⚠️ Pièges QCM · Séance 8

- RNN : **mêmes poids** à chaque pas ; paramètres indépendants de la longueur.
- Paramètres `nn.RNN` : $n_h \cdot n_\text{in} + n_h \cdot n_h + 2n_h$ ; GRU ×3 ; LSTM ×4.
- `nn.LSTM` renvoie `out, (h_n, c_n)` ; `nn.RNN` renvoie `out, h_n`. `h_n` a la shape `(num_layers, batch, hidden)`.
- Explosion → **clipping** ; vanishing → **LSTM/GRU** (chemin additif).
- Portes du LSTM : **sigmoïdes** ; candidat et sortie : **tanh**.
- Attention : softmax des scores → poids **positifs qui somment à 1** ; contexte = moyenne pondérée.
- Multiplier les scores → attention plus **piquée** (pas "plus uniforme").
- Transformer : parallèle, dépendances longues en un pas, mais beaucoup de données.
- Cosinus ∈ **[−1, 1]**.

---

## 9. Formulaire express

### Les nombres magiques

| Valeur | À quoi ça sert |
| --- | --- |
| $\sigma(0) = 0.5$ | la frontière de décision |
| $\max\sigma' = 0.25$ | vanishing gradient des sigmoïdes |
| $\tanh'(0) = 1$, $\text{ReLU}' = 1$ (si $z > 0$) | pourquoi ReLU et tanh transmettent mieux |
| $\ln 2 = 0.693$ | log loss d'une prédiction 0.5 |
| $\ln 10 = 2.303$ | loss initiale à 10 classes ; loss "plate" = rien n'est appris |
| $e \approx 2.718$, $e^2 \approx 7.389$, $e^{-2} \approx 0.135$ | softmax et sigmoïde à la main |
| $(0.665, 0.245, 0.090)$ | softmax de $(2, 1, 0)$ |
| $1/(1-\beta) = 10$ | momentum $\beta = 0.9$ |
| $1/(1-p)$ | facteur du dropout en train |

### Les formules

| Thème | Formule |
| --- | --- |
| Perceptron | $\hat y = 1$ si $w^\top x + b \ge 0$ ; $w \leftarrow w + \eta(y - \hat y)x$, $b \leftarrow b + \eta(y - \hat y)$ |
| Sigmoïde | $\sigma(z) = 1/(1 + e^{-z})$, $\sigma' = \sigma(1 - \sigma)$, $\sigma(-z) = 1 - \sigma(z)$ |
| Log loss | $J = -\frac{1}{m}\sum[y\log a + (1-y)\log(1-a)]$ |
| Gradient logistique | $dw = \frac{1}{m}X^\top(A - y)$, $db = \frac{1}{m}\sum(a - y)$ |
| Descente de gradient | $w \leftarrow w - \eta\,\partial J/\partial w$ |
| Paramètres dense | $n_\text{out}(n_\text{in} + 1)$ |
| Backprop | $dZ^{[L]} = A^{[L]} - Y$ ; $dZ^{[l]} = W^{[l+1]\top}dZ^{[l+1]} \odot \sigma'(Z^{[l]})$ ; $dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]\top}$ |
| Softmax | $p_k = e^{z_k}/\sum_j e^{z_j}$ |
| Cross-entropy | $L = -\log p_\text{vraie}$, $\partial L/\partial z = p - y$ |
| Pas par epoch | $\lceil m/B \rceil$ |
| Init | Xavier $1/n_\text{in}$ ou $2/(n_\text{in} + n_\text{out})$ ; He $2/n_\text{in}$ |
| Weight decay | $w \leftarrow (1 - \eta\lambda)w - \eta g$ |
| Dropout | $\tilde a = m\,a/(1-p)$ en train, $a$ en eval |
| Batch norm | $\hat z = (z - \mu_B)/\sqrt{\sigma_B^2 + \varepsilon}$, $y = \gamma\hat z + \beta$ ; $\mu_\text{run} \leftarrow 0.9\mu_\text{run} + 0.1\mu_B$ |
| Momentum | $v \leftarrow \beta v + g$, $w \leftarrow w - \eta v$ |
| Adam | $\hat m = m/(1-\beta_1^t)$, $\hat s = s/(1-\beta_2^t)$, $w \leftarrow w - \eta\hat m/(\sqrt{\hat s} + \varepsilon)$ |
| Clipping | si $\|g\| > c$ : $g \leftarrow c\,g/\|g\|$ |
| Conv : taille | $\lfloor(n + 2p - k)/s\rfloor + 1$ ; "same" : $p = (k-1)/2$ |
| Conv : paramètres | $C_\text{out}(C_\text{in}k^2 + 1)$ |
| Champ récepteur | $r \leftarrow r + (k-1)j$, $j \leftarrow j \cdot s$ (départ $r = j = 1$) |
| ResNet | $y = F(x) + x$, $\partial y/\partial x = \partial F/\partial x + I$ |
| RNN | $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$ |
| LSTM | $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, $h_t = o_t \odot \tanh(c_t)$ |
| Attention | $e_i = s^\top h_i$, $\alpha = \text{softmax}(e)$, $c = \sum\alpha_i h_i$ ; self : $\text{softmax}(QK^\top/\sqrt d)V$ |
| Cosinus | $u^\top v/(\|u\|\|v\|) \in [-1, 1]$ |

### Les dates et les noms (au cas où)

McCulloch et Pitts 1943 · Rosenblatt (perceptron) 1958 · Novikoff 1962 · Rumelhart, Hinton, Williams (backprop) 1986 · Hochreiter et Schmidhuber (LSTM) 1997 · LeNet-5 1998 · Glorot / Xavier 2010 · Nair et Hinton (ReLU) 2010 · AlexNet 2012 · VGG 2014 · GRU (Cho) 2014 · Dropout (Srivastava, Hinton) 2014 · Seq2seq (Sutskever) 2014 · Batch norm (Ioffe, Szegedy) 2015 · Adam (Kingma, Ba) 2015 · He / Kaiming 2015 · ResNet 2015 · Attention (Bahdanau) 2015 · Transformer (Vaswani) 2017 · FashionMNIST 2017.

---

## 10. Le code PyTorch à savoir lire

**Le modèle**

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()                  # toujours en premier
        self.hidden = nn.Linear(64, 32)     # weight (32, 64), bias (32,)
        self.out = nn.Linear(32, 10)        # weight (10, 32), bias (10,)
    def forward(self, x):                   # x : (batch, 64)
        a1 = torch.sigmoid(self.hidden(x))
        return self.out(a1)                 # logits (batch, 10), pas de softmax
```

**Compter les paramètres** : `sum(p.numel() for p in model.parameters())`.

**Autograd à la main**

```python
w = torch.tensor([2.0, -1.0], requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)
x, y = torch.tensor([1.0, 3.0]), torch.tensor(1.0)
a = torch.sigmoid(w @ x + b)                # 0.1192
loss = -(y * torch.log(a) + (1 - y) * torch.log(1 - a))
loss.backward()
w.grad   # (a - y) * x = (-0.8808, -2.6424)
```

**Mise à jour manuelle**

```python
with torch.no_grad():
    w -= lr * w.grad
w.grad.zero_()
```

**Évaluation**

```python
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total
```

**Early stopping**

```python
if val_loss < best_loss:
    best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
else:
    wait += 1
    if wait >= patience:
        break
model.load_state_dict(best_state)
```

**Optimiseurs et schedulers**

```python
torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)
torch.optim.Adam(model.parameters(), lr=1e-3)
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
scheduler.step()     # une fois par epoch, après les optimizer.step()
```

**Initialisation**

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

**Sauvegarde et device**

```python
torch.save(model.state_dict(), "model.pt")
model2 = MLP(); model2.load_state_dict(torch.load("model.pt")); model2.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device); xb, yb = xb.to(device), yb.to(device)
```

**Gradient accumulation (+ mixed precision)**

```python
accum = 4                                    # batch effectif = 4 x batch
optimizer.zero_grad()
for step, (xb, yb) in enumerate(train_loader):
    with torch.autocast(device_type=device, dtype=dtype):
        loss = loss_fn(model(xb), yb) / accum
    scaler.scale(loss).backward()            # les gradients S'ADDITIONNENT
    if (step + 1) % accum == 0:              # tous les 4 batchs : un pas
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Les erreurs que l'examen peut glisser dans du code**

| Erreur | Conséquence |
| --- | --- |
| `softmax` avant `CrossEntropyLoss` | double softmax, apprentissage dégradé |
| `sigmoid` avant `BCEWithLogitsLoss` | double sigmoïde |
| labels `float` / one-hot dans `CrossEntropyLoss` | erreur ou mauvaise loss : il faut des int64 `(B,)` |
| pas de `zero_grad()` | gradients accumulés, pas faux |
| pas de `model.eval()` | dropout/batch norm en mode train pendant l'évaluation |
| pas de `torch.no_grad()` en éval | lent, consomme de la mémoire (pas faux) |
| évaluer sur `train_loader` | score optimiste |
| pas d'activation entre deux `nn.Linear` | modèle linéaire |
| `model.forward(x)` au lieu de `model(x)` | mauvaise habitude (saute les hooks) |
| modèle et données sur des devices différents | `RuntimeError` |
| normaliser avec les stats de tout le dataset | fuite de données (leakage) |
| `scheduler.step()` dans la boucle des batchs (avec `T_max = n_epochs`) | le lr décroît trop vite |

---

## 11. Problème → solution : la grande table

| Problème | Solution | Pourquoi ça marche |
| --- | --- | --- |
| XOR / frontière non linéaire | couches cachées + activation non linéaire | chaque neurone caché trace une droite, la sortie les combine |
| Pas de proba, pas de gradient | sigmoïde + log loss | dérivable, convexe, $(a-y)x$ |
| Données non séparables, pas de "meilleure" droite | minimiser une loss | un objectif clair à optimiser |
| Vanishing gradient (sigmoïdes profondes) | **ReLU** | dérivée 1 sur les chemins actifs |
| Neurones identiques | init **aléatoire** | casse la symétrie |
| Signal qui disparaît / explose à l'init | **Xavier** (tanh) / **He** (ReLU) | garde la variance constante de couche en couche |
| Dead ReLU | Leaky ReLU, lr plus petit, He | gradient jamais exactement nul |
| Overfitting | plus de données, **weight decay**, **dropout**, **early stopping** | poids petits, redondance, arrêt au bon moment |
| Entrées de couches mal échelonnées | **batch norm** (CNN) / **layer norm** (transformers, RNN) | entrées bien normalisées → lr plus grands, moins sensible à l'init |
| SGD qui zigzague / rampe | **momentum**, **Adam** | élan + pas adapté à chaque paramètre |
| Loss qui explose / nan | lr ÷ 10, **gradient clipping** | limite la taille des pas |
| Trop de paramètres sur les images, pas de voisinage | **convolution** | localité + partage des poids |
| Dégradation des réseaux très profonds | **ResNet** ($y = F(x) + x$) | chemin identité pour le gradient |
| Peu de données d'images | **transfer learning** | réutilise les features apprises sur ImageNet |
| Séquences de longueur variable | **RNN** | mêmes poids à chaque pas de temps |
| Vanishing dans le temps | **LSTM / GRU** | cell state additif, porte forget ≈ 1 |
| Explosion dans le temps | **gradient clipping** | limite la norme |
| Tout dans un seul vecteur (encoder-decoder) | **attention** | relire tous les états à chaque pas de sortie |
| Calcul séquentiel, dépendances longues | **transformer** (self-attention) | tout en parallèle, toutes les positions reliées |
| GPU trop petit | **gradient accumulation**, **mixed precision** | même gradient qu'un gros batch ; 2× plus rapide, moitié de mémoire |
