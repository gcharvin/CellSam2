# Synthese: Parentage Bud->Mere (etat au 2026-04-08)

Cette note consolide l'historique complet du developpement du parentage
`bud -> mother` pour la levure bourgeonnante dans `CellSam2`.
Elle remplace les notes partielles anterieures comme point de depart
pour un agent ou un developpeur futur.

Notes de detail:
- modeles et scores: `NOTE_MODELES_PARENTAGE_AGENT.md`
- protocole d'experiment: `EXPERIMENTS.md`
- infra serveur et workflow: `CLAUDE.md`

---

## Le probleme

Le tracker SAM2 produit des tracklets mais ne sait pas quelle cellule
est la mere d'un bud. Le parentage est assigne en post-traitement.

Specificites de la levure bourgeonnante par rapport aux bacteries:
- division asymetrique: un petit bud emerge d'une mere specifique
- une mere ne peut pas budder deux fois en moins de 8 frames (periode refractaire)
- un bud peut apparaitre equidistant de plusieurs meres (ambiguite)
- les erreurs residuelles sont des confusions **locales de genealogie**:
  - ancetre choisi a la place de la mere
  - descendante recente choisie comme mere
  - soeur proche dans la meme cavite

---

## Historique des approches

### Phase 1 — Heuristique globale

**Modele:** `hybrid-cost-v1`
**Principe:** score heuristique geometrique + ILP global
**Features:** distance, contact, neck, taille, mouvement, angle, maturite
**Val F1:** 0.662 / **DivA:** 8.83

Conclusion: robuste et interpretable, mais plafonne sur les conflits locaux.

### Phase 2 — Apprentissage pairwise sans SAM2

**Modele:** `pairwise-temporal-v1`
**Principe:** regression logistique sur features heuristiques + temporelles
**Val F1:** 0.641 — **pire** que la baseline sur holdout (sur-apprentissage)

### Phase 3 — Apprentissage pairwise avec SAM2

**Modele:** `pairwise-temporal-sam2-v1`
**Principe:** memes features + similarite cosinus d'embeddings SAM2 (objet entier)
**Val F1:** 0.654 / **diag:** correct=40, wrong=9

### Phase 4 — Meilleur modele actuel

**Modele:** `pairwise-temporal-sam2-blend060-v1`
**Principe:** score appris blende avec l'heuristique (alpha=0.6) + ILP
**Val F1:** 0.662 / **DivA:** 10.28 / **diag:** correct=41, wrong=9, orphan=2

Experiment de reference:
```
/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1455_ce88ca0_pairwise-temporal-sam2-blend060-v1
```

### Phase 5 — Approches plus complexes (toutes inferieures)

| Modele | Val F1 | Conclusion |
|--------|--------|------------|
| Transformer listwise | 0.615-0.619 | Sur-apprentissage |
| ContextRanker v2 | 0.637 | Bon protocole, sous-performe |
| ContextRanker v3 (reg) | 0.624 | Regularisation casse la capacite utile |
| ContextRanker v4 (attn) | 0.624 | Attention inter-candidates sans gain |

Lecon: augmenter la capacite du modele ne progresse pas avec ~190 events
de training. Changer les features a progresse; changer l'architecture non.

### Phase 6 — Features SAM2 neck (tente, rejete)

**Modele:** `pairwise-temporal-sam2-neck-blend060-v1`
**Hypothese:** pooler les embeddings SAM2 sur la region neck (interface
mere-bud) plutot que sur l'objet entier devrait discriminer
ancetre/mere/descendante via la morphologie d'isthme.
**Nouvelles features:** `sam2_neck_cosine_mean`, `sam2_neck_cosine_min`,
`sam2_neck_valid_fraction` (26 features au total)
**Val F1:** 0.649 / **diag:** correct=38, wrong=11 — **pire que la reference**

Experiment:
```
/home/charvin-admin/Documents/cellSAM2/experiments/20260406_0942_d226da0_pairwise-temporal-sam2-neck-blend060-v1
```

Cause de l'echec: la similarite cosinus SAM2 sur la region neck est trop
peu specifique. Un ancetre adjacent partage le meme voisinage spatial que
la vraie mere — le signal ne discrimine pas l'isthme d'emergence de la
simple proximite geometrique.

---

## Analyse des erreurs residuelles

Sur `val_holdout`, le meilleur modele fait 9 `stable_wrong_parent`:

```
ancestor_of_true  = 4
descendant_of_true = 3
sibling_of_true   = 1
other_branch      = 1
```

Dans 8/9 cas, le modele trouve le bon voisinage spatial. Il echoue
sur la discrimination genealogique locale.

**Caveat important:** certaines de ces 9 erreurs sont probablement des
cas biologiquement ambigus (bud detecte en retard, timing shift,
hallucination). Le vrai plafond atteignable est inconnu sans revue
experte video par video.

---

## Etat du code

### API standardisee

`tools/parentage_api.py` — point d'entree unique pour tous les scorers:

```python
build_candidates(seq_dir, cfg)          # candidates + features geometriques
build_model_inputs(candidate_table, scorer_name, cfg, embedding_extractor)
score_candidates(model_inputs, scorer, cfg)
assign_parentage(candidate_table, scored_candidates, cfg, mode)
write_assignments(assignment, res_track_path)
```

Scorers disponibles: `HeuristicScorer`, `LinearReranker`,
`TransformerReranker`, `ContextReranker`.

### Features actuelles (26)

13 heuristiques + 7 temporelles + 3 SAM2 globales + 3 SAM2 neck.
Definies dans `FEATURE_NAMES` dans `tools/learned_bud_rerank.py`.

Note: les 3 features neck sont presentes dans le code mais **ne doivent
pas etre retrainee** — leur modele associe est inferieur a la reference.
Un retrain avec uniquement les 23 features de la reference produira le
meilleur modele.

### Script de lancement

`run_neck_experiment.sh` — sert de template pour tout nouvel experiment
pairwise. Adapte le tag et les features, le reste est reutilisable.

---

## Ce qui reste ouvert

### Priorite 1 — Qualifier le vrai plafond (1 jour)

Regarder les 9 `stable_wrong_parent` du meilleur modele sur video
(`render_parentage_comparison.py`). Determiner combien sont:
- de vraies erreurs corrigeables
- des cas ambigus meme a l'oeil expert

Sans cette information, tout effort technique est aveugle.

### Priorite 2 — Features genealogiques (si des erreurs sont corrigeables)

Deux features ciblent directement les confusions ancetre/descendante
sans toucher a SAM2 et sans plus de donnees:

**`frames_since_last_bud`** — nombre de frames depuis le dernier
bourgeonnement du candidat-mere. Une cellule qui a elle-meme emerge
recemment est probablement une descendante, pas une mere. Le
`mother_age` actuel compte depuis l'apparition du track, pas depuis
le dernier bourgeonnement. Calculable depuis `track_infos` et
`res_track` (~15 lignes).

**`lineage_depth`** — profondeur de genealogie du candidat dans l'arbre.
Un ancetre est par definition a >= 2 niveaux. `_is_ancestor()` existe
deja dans `tools/online_bud_parentage.py` — extraire la profondeur
plutot qu'un booleen (~5 lignes).

### Priorite 3 — Donnees (si les features ne suffisent pas)

Avec ~190 events en training, tout modele avec plus de ~100 parametres
sur-apprend. Annoter 5-10 videos supplementaires doublerait le dataset
et rendrait le ContextRanker ou un Set Transformer envisageables.

Le format d'annotation est `man_track.txt` — le label est derive
automatiquement par matching IoU, aucune annotation manuelle des paires
n'est necessaire.

### Priorite 4 — Evaluation applicative (parallele)

Les metriques actuelles (`division_iou_window`, `DivA`) penalisent
les decalages de timing et les buds discutables de facon uniforme.
Une evaluation ciblee `bas de cavite` avec tolerance au timing serait
plus representative de la qualite biologique percue.

---

## Ce qu'il ne faut pas faire

- Ne pas pousser de nouvelles variantes SAM2 cosinus (global ou neck):
  le signal est trop peu specifique pour discriminer la genealogie locale.
- Ne pas augmenter la capacite du transformer sans plus de donnees.
- Ne pas optimiser uniquement `division_iou_window F1` sans revue expert:
  il penalise des cas biologiquement acceptables.
- Ne pas modifier `mode=online` ou `mode=hybrid` sans verifier qu'ils
  utilisent bien l'API standardisee (`parentage_api.py`).

---

## Commits recents (session 2026-04)

| Commit | Description |
|--------|-------------|
| `605219e` | Document neck SAM2 experiment result (rejected) |
| `e253956` | Add run_neck_experiment.sh launch script |
| `d226da0` | Add SAM2 neck-region embedding features (23->26) |
| `da6eeb1` | Add Claude handoff note (CLAUDE.md) |
| `1b498b5` | Route heuristic global parentage through shared API |
| `eea2f17` | Route parentage CLIs through shared API |
| `8a0d0a6` | Add standardized parentage scoring API |
