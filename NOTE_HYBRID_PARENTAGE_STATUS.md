# Note de situation: solveur hybride de parentage

Associe au commit `2ecbdd9` (`Add hybrid bud parentage optimization`)

## Objet

Cette note resume la situation apres l'ajout d'un solveur hybride de parentage `bud -> mother`, qui combine:

- les propositions locales deja produites par le post-traitement `online-post`;
- une re-optimisation globale uniquement sur les cas ambigus, sous contraintes temporelles.

## Ce qui a ete ajoute

Le commit `2ecbdd9` introduit un mode `hybrid` dans:

- `tools/online_bud_parentage.py`
- `inference/track_cells.py`

Principe:

1. partir des assignations `online-post` existantes;
2. verrouiller les assignations non conflictuelles et suffisamment confiantes;
3. re-optimiser globalement les buds restants avec un solveur binaire sous contraintes:
   - un seul parent par bud;
   - periode refractaire par mere;
   - cout base sur distance, taille, mouvement, et eventuellement contact.

Le meilleur reglage teste est:

- `proposal_bonus = 0.05`
- `proposal_lock_score = 0.60`

## Resultat principal

Run de reference:

- `/home/charvin-admin/Documents/cellSAM2/experiments/20260311_1846_d63d81d_hybrid-parentage-v2`

Resume:

- `division_iou_window_f1 = 0.658`
- `precision = 0.538`
- `recall = 0.847`
- `Cell-HOTA HOTA = 36.594`
- `Cell-HOTA DivA = 8.553`

## Comparaison aux baselines

Baseline `no-post`:

- `F1 = 0.635`
- `HOTA = 36.694`
- `DivA = 8.764`

Baseline `online-post`:

- `F1 = 0.649`
- `HOTA = 35.522`
- `DivA = 7.838`

Global solver pur `no-post -> global`:

- `F1 = 0.610`
- `HOTA = 36.974`
- `DivA = 9.018`

Hybrid solver `online-post -> hybrid`:

- `F1 = 0.658`
- `HOTA = 36.594`
- `DivA = 8.553`

## Interpretation

Le solveur hybride est la meilleure variante testee a ce stade:

- il ameliore le `F1` division par rapport a `online-post`;
- il ameliore aussi `HOTA` et `DivA` par rapport a `online-post`;
- il evite la degradation observee quand on remplace completement les assignations locales par un solveur global.

En revanche, le gain reste modeste. Ce commit valide donc surtout une direction:

- l'optimisation globale du parentage aide reellement;
- mais elle ne suffit pas, a elle seule, a produire un saut de performance.

## Conclusion pratique

Apres `2ecbdd9`, la meilleure baseline de post-traitement est le mode `hybrid`.

Mais la situation reste la suivante:

- le verrou principal est toujours le parentage;
- on a probablement extrait l'essentiel du gain "combinatoire simple";
- pour un gain plus net, il faudra enrichir le score d'association ou apprendre explicitement l'association `bud -> mother`.

## Suite recommandee

Priorite 1:

- analyser les cas residuels `stable_wrong_parent` du mode `hybrid`.

Priorite 2:

- enrichir le cout global avec des indices plus discriminants:
  - direction locale de croissance;
  - signal de col/bud-neck;
  - competition spatiale explicite entre meres voisines;
  - historique local avant emergence.

Priorite 3:

- si ces enrichissements restent insuffisants, passer a un module d'association appris, plus `bud-centric`.
