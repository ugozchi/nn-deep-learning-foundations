# TP01 : le perceptron en NumPy, explications

## Idée générale

Un perceptron calcule un score `z = w·x + b`, puis prédit `1` si `z ≥ 0`, sinon `0`. Géométriquement, c'est une **droite** qui sépare le plan en deux. L'apprentissage consiste à déplacer cette droite à chaque erreur.

## Partie A : prédire

| Ex. | Code | Pourquoi |
|---|---|---|
| 1 | `np.asarray(z >= 0).astype(int)` | Compare puis convertit `True/False` en `1/0`. `asarray` permet de gérer aussi un simple nombre. |
| 2 | `step(w @ x + b)` | Score puis seuil. |
| 3 | Tables de vérité en tableaux | `X_table` (4×2) et les labels AND / OR / XOR. |
| 4 | `step(X @ w + b)` | Même chose sur toutes les lignes d'un coup (broadcasting de `b`). |
| 5 | OR : `w=(1,1), b=-0.5` ; NAND : `w=(-1,-1), b=1.5` | OR : on baisse le seuil. NAND : on inverse tous les signes de AND (`w=(1,1), b=-1.5`). |

## Partie B : apprendre

- **Ex. 6, `update`** : `e = y - ŷ`, puis `w ← w + η·e·x` et `b ← b + η·e`.
  - Si la prédiction est juste, `e = 0` et rien ne change.
  - Faux positif : `e = -1`, on retire `x` de `w`.
  - Faux négatif : `e = +1`, on ajoute `x` à `w`.
  - On écrit `w = w + ...` (pas `+=`) pour ne pas modifier le tableau de l'appelant.
- **Ex. 7, `count_mistakes`** : `np.sum(predict(X, w, b) != y)`.
- **Ex. 8, `train_perceptron`** : on part de `w=0, b=0`. À chaque epoch, on parcourt les exemples dans l'ordre. On compte une erreur avant chaque `update`, on stocke le total de l'epoch, et on arrête dès qu'une epoch n'a **aucune** erreur.
- **Ex. 9 et 10** : on appelle `train_perceptron` sur OR, puis sur les 4 points du cours. On retrouve bien `[3, 1, 0]` erreurs et la droite `x₁ + x₂ - 2 = 0`.

## Partie C : un vrai jeu de données 2D

- **Ex. 11** : entraînement sur deux nuages séparables. Le nombre d'erreurs tombe à 0 en 5 epochs.
- **Ex. 12, `accuracy`** : `np.mean(y_true == y_pred)`. La moyenne de booléens donne la fraction de `True`.
- **Ex. 13, `line_x2`** : on isole `x₂` dans `w₁x₁ + w₂x₂ + b = 0`, ce qui donne `x₂ = -(w₁x₁ + b) / w₂`. C'est ce qui sert à tracer la droite.
- **Ex. 14, learning rate** : avec `η = 0.1`, `w` et `b` sont 10 fois plus petits mais les erreurs par epoch sont identiques. Partant de `w = 0`, tout est multiplié par `η`, donc `z` aussi, et le `step` ne regarde que le **signe** de `z`. Pour le perceptron, `η` n'a aucun effet (il en aura avec la descente de gradient).

## Partie D : les limites

- **Ex. 15, XOR** : le nombre d'erreurs ne tombe jamais à 0, car aucune droite ne sépare XOR.
- **Ex. 16, nuages qui se chevauchent** : les erreurs montent et descendent sans fin (accuracy 0.76). La droite finale dépend simplement du moment où on s'arrête.
- **Ex. 17, pente du step** : la différence finie `(step(z+h) - step(z-h)) / 2h` vaut 0 partout (sauf en 0). Il n'y a donc pas de gradient.
- **Ex. 18, les 4 limites**, chacune avec sa solution dans les prochaines séances :
  1. Une seule droite, donc pas de XOR → plusieurs couches de neurones (séance 3).
  2. Données non séparables : la règle ne s'arrête jamais → minimiser une loss (séance 2).
  3. Sortie 0/1, aucune confiance → sigmoïde, donc une probabilité (séance 2).
  4. Dérivée nulle, donc pas de gradient → sigmoïde dérivable, donc descente de gradient (séance 2).

## Bonus

- **Ex. 19, classe `Perceptron`** : `fit` appelle `train_perceptron` et stocke `w`, `b`, `mistakes`, et `predict` réutilise la fonction `predict`. C'est le style scikit-learn.
- **Ex. 20, scikit-learn** : même précision (1.0) mais poids différents. scikit-learn mélange les exemples à chaque epoch, et plusieurs droites séparent bien deux nuages éloignés.

## Vérification

J'ai exécuté toutes les cellules dans l'ordre : tous les `assert` passent et chaque exercice affiche `Exercise validated!`.
