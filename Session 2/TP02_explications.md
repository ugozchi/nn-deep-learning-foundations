# TP02 : régression logistique, de NumPy à PyTorch, explications

## Idée générale

On remplace le `step` du perceptron par la **sigmoïde** `σ(z) = 1 / (1 + e^-z)`. Le modèle produit alors une **probabilité** `a = σ(w·x + b)` au lieu d'un 0/1. On mesure l'erreur avec la **log loss** et on minimise cette loss par **descente de gradient**. La sigmoïde est dérivable, donc le gradient existe (c'était la limite n°4 du perceptron).

Le TP implémente tout à la main en NumPy (parties A et B), puis refait la même chose avec PyTorch (partie C).

## Cellules de mise en place

| Cellule | Rôle |
|---|---|
| Imports | `numpy` (modèle des parties A/B), `matplotlib` (courbes, frontière), `torch` / `torch.nn` (partie C), `make_blobs` et `make_moons` (jeux de données 2D). `FAST_RUN` n'a pas d'effet ici, tout est rapide. |
| `np.random.seed(0)`, `torch.manual_seed(0)` | Résultats reproductibles. Les `assert` dépendent de cette graine. |
| `plot_boundary` | Colore le plan avec `P(y=1\|x)` (`contourf`) et trace la frontière `P = 0.5` (`contour`). Elle prend n'importe quelle fonction `predict_proba`. |
| `make_blobs` | 100 « plantes » à 2 mesures, 2 nuages (toxique = 1, inoffensive = 0). `X` : `(100, 2)`, `y` : `(100,)`. |

## Partie A : la régression logistique en NumPy

| Ex. | Code | Pourquoi |
|---|---|---|
| 1, `sigmoid` | `1 / (1 + np.exp(-z))` | `np.exp` travaille élément par élément, donc ça marche sur un nombre comme sur un tableau (`math.exp` ne le ferait pas). |
| 2, `initialisation` | `np.random.randn(n), np.random.randn()` | `w` suit une loi normale `N(0,1)`, `b` aussi. L'ordre compte : on tire `w` d'abord, puis `b`, sinon le test sur la graine échoue. |
| 3, `model` | `sigmoid(X @ w + b)` | `X @ w` : `(m, n) @ (n,) → (m,)`, un score par exemple. `b` est ajouté à chaque ligne (broadcasting), puis la sigmoïde donne les `m` probabilités, sans boucle. |
| 4, `log_loss` | `-np.mean(y*np.log(A+eps) + (1-y)*np.log(1-A+eps))` | `J = -1/m Σ [y log a + (1-y) log(1-a)]`. Pour `y=1` seul le premier terme reste (pénalise un `a` proche de 0), pour `y=0` seul le second. `eps = 1e-15` évite `log(0) = -inf`. |
| 5, `gradients` | `dw = X.T @ (A - y) / len(y)` ; `db = np.mean(A - y)` | `X.T` est `(n, m)`, `(A - y)` est `(m,)`, donc `dw` est `(n,)`. C'est la formule du cours : l'erreur `(a - y)` pondérée par chaque feature, moyennée. |
| 6, `update` | `w - eta*dw`, `b - eta*db` | Un pas de descente de gradient (on va **contre** le gradient). Avec `η = 2` sur l'exemple du cours : `w = (1, -0.5)` et la loss passe de 0.693 à 0.257. |
| 7, `logistic_regression` | boucle `n_iter` fois : `model` → `log_loss` (stockée dans `losses`) → `gradients` → `update` | On stocke la loss **avant** la mise à jour. `losses` décroît strictement (pas de `η` trop grand) et finit sous 0.2 (0.148 observé). |
| 8, `predict` + `acc` | `(model(X,w,b) >= 0.5).astype(int)` ; `np.mean(predict(X,w,b) == y)` | Seuil à 0.5 pour passer de probabilité à classe, puis la moyenne de booléens donne la fraction de bonnes réponses : **0.93**. |
| 9, `p_new` | `model(x_new, w, b)[0]` | `model` attend un tableau 2D `(1, 2)` et renvoie un tableau de taille 1, d'où `[0]`. Résultat : `P(toxique) ≈ 0.969`. |

**Ce qu'on voit.** La loss chute vite puis lentement. La frontière est une droite (`w·x + b = 0`), mais chaque point a maintenant une probabilité : la couleur s'estompe près de la droite, là où le modèle hésite.

## Partie B : vérifier le gradient numériquement

Une formule dérivée à la main peut être fausse. Le test classique est la **différence finie centrée** : `∂J/∂wⱼ ≈ (J(wⱼ+h) - J(wⱼ-h)) / 2h`.

- **Ex. 10, `numerical_gradient`** : pour chaque `j`, on copie `w` deux fois (`w_plus`, `w_minus`), on ajoute `+h` / `-h` à la composante `j`, et on calcule la différence des deux loss divisée par `2h`. Il faut travailler sur des **copies** : modifier `w` en place fausserait les composantes suivantes (le test vérifie que `w` n'est pas modifié).
- **Ex. 11, erreur relative** : `‖dw_formule - dw_num‖ / (‖dw_formule‖ + ‖dw_num‖)` avec `np.linalg.norm`. On obtient **≈ 4e-11**, donc la formule `(a - y)·x` est correcte. L'erreur relative est préférable à l'erreur absolue car elle ne dépend pas de l'échelle du gradient.

## Partie C : la même chose en PyTorch

Un **tenseur** PyTorch est un tableau NumPy qui peut aller sur GPU et **enregistrer les opérations** pour calculer les gradients automatiquement (autograd).

- **Ex. 12, tenseurs** : `torch.tensor(X, dtype=torch.float32)`, pareil pour `y`. `float32` est le type par défaut des couches PyTorch, et `y` doit aussi être un flottant pour la loss.
- **Ex. 13, autograd** : on crée `w_t`, `b_t` avec `requires_grad=True`, on écrit le forward (`torch.sigmoid(X_t @ w_t + b_t)`) et la loss avec `torch.log` et `.mean()` (même formule qu'en NumPy, sans `eps`), puis `loss.backward()` remplit `w_t.grad` et `b_t.grad` par la règle de la chaîne. Les gradients sont identiques à ceux de la formule : `[-0.1347, 0.5785]`.
- **Ex. 14, descente à la main avec autograd** : à chaque itération : forward, loss, `backward()`, puis mise à jour **dans `with torch.no_grad():`** (sinon la mise à jour elle-même serait enregistrée), et enfin `.grad.zero_()` car PyTorch **additionne** les nouveaux gradients aux anciens. Partant des mêmes valeurs initiales que l'exercice 7, on retrouve les mêmes poids (`[1.084, -1.878]`, `b = 3.142`).
- **Ex. 15, le modèle idiomatique** :
  - `nn.Linear(2, 1)` : couche qui contient `weight` `(1, 2)` et `bias` `(1,)` et calcule les **logits** `z`.
  - `nn.BCEWithLogitsLoss()` : la log loss, qui prend les logits et applique la sigmoïde elle-même de façon stable.
  - `torch.optim.SGD(model_t.parameters(), lr=0.1)` : fait `w ← w - η∇J` pour tous les paramètres.
- **Ex. 16, la boucle en 5 lignes** : `zero_grad()` → `z = model_t(X_t).squeeze(1)` → `loss = loss_fn(z, y_t)` → `loss.backward()` → `optimizer.step()`. `squeeze(1)` enlève la dimension de taille 1 pour avoir `(100,)` comme `y_t`. On stocke `loss.item()` (un nombre Python). Résultat : loss 0.156, accuracy 0.92. Positif `z > 0` équivaut à `a > 0.5`, d'où le test `z > 0` pour l'accuracy. Les poids diffèrent un peu de NumPy (autre départ aléatoire) mais c'est le même algorithme.
- **Ex. 17, pourquoi des logits** : avec `z = -200` et `y = 1`, `-log(sigmoid(z))` donne **inf** : en float32, `sigmoid(-200)` s'arrondit à 0 et `-log(0) = inf`. `BCEWithLogitsLoss` donne **200**, la vraie valeur `log(1 + e^200)`, car elle ne passe jamais par la probabilité (astuce du log-sum-exp).

## Partie D : la limite

- **Ex. 18, les lunes** : sur `make_moons`, la régression logistique (`η = 1.0`, 1000 itérations) atteint **0.885**. Elle trouve la meilleure **droite**, mais les pointes des lunes restent du mauvais côté. Entraîner plus longtemps n'aide pas : la limite vient du **modèle** (une frontière linéaire), pas de l'optimisation. C'est ce que la séance 3 corrigera avec plusieurs neurones.

## Pour aller plus loin

- **Ex. 19, learning rate** : on entraîne avec `η ∈ {0.001, 0.1, 10}` pendant 200 itérations, en remettant `np.random.seed(0)` avant chacun pour avoir le même départ. Avec `0.001`, la loss bouge à peine (trop lent). Avec `10`, les pas sont si grands que la loss remonte à certaines itérations (le pas dépasse le minimum). `0.1` est stable et rapide. Contrairement au perceptron, ici **`η` compte**, et on le règlera dans chaque projet.

## Vérification

J'ai complété les zones `START / END CODE HERE` et exécuté toutes les cellules dans l'ordre : tous les `assert` passent et chaque exercice affiche `Exercise validated!`. Le notebook contient donc les solutions et les sorties.
