# Note Agent: Modeles De Parentage Bud->Mere

## Objet

Cette note resume les modeles de parentage testes pour la levure dans `CellSam2`, leurs entrees, leur type d'apprentissage, leurs forces/limites, et les scores obtenus. Elle sert de point de reprise pour un agent futur.

Contexte:
- tracking et segmentation SAM2 deja produits en amont
- parentage `bud -> mother` assigne a posteriori
- optimisation globale finale via ILP avec contrainte de refractory period
- split courant:
  - `train_dev`: videos `01-11`
  - `val_holdout`: videos `12-14`

Metrique principale de travail:
- `division_iou_window` sur `val_holdout`

Metriques complementaires:
- diagnostic bud-centric: `stable_correct_parent`, `stable_wrong_parent`, `stable_orphan`
- Cell-HOTA quand calcule: `HOTA`, `DivA`, `DivRe`, `DivPr`

## Resume Court

Le meilleur compromis actuel est le reranker appris `pairwise + SAM2 + blend060`.

Pourquoi:
- meilleur ou quasi meilleur `division_iou_window F1` sur `val_holdout`
- meilleur parentage bud-centric parmi les modeles evalues proprement
- meilleur `DivA` Cell-HOTA parmi les runs calcules
- complexite encore raisonnable

Limite principale restante:
- erreurs locales entre candidates tres proches dans une structure dense
- surtout confusions `ancetre / vraie mere / descendante recente`

## Modeles

### 1. Baseline heuristique globale

Nom court:
- `hybrid-cost-v1`

Principe:
- generation de candidates `bud -> mother`
- score heuristique par candidate
- optimisation globale ILP sous contraintes temporelles

Input:
- tracklets deja produites
- features geometriques et temporelles simples:
  - distance
  - ratio de taille
  - mouvement
  - contact
  - neck
  - angle
  - maturite
  - qualite de tracklet
  - marge

Training:
- aucun

Forces:
- simple
- robuste avec peu de donnees
- facilement interpretable

Limites:
- plafonne vite dans les cas ambigus
- peu de signal appris
- resout mal certains conflits locaux mere/fille/ancetre

Run de reference:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1130_854f313_hybrid-cost-v1-postproc-dev-holdout`

Scores:
- `train_dev` F1: `0.681223`
- `val_holdout` F1: `0.662338`
- `val_holdout` diagnostic:
  - `stable_correct_parent = 32`
  - `stable_wrong_parent = 15`
  - `stable_orphan = 5`
- Cell-HOTA:
  - `HOTA = 36.809`
  - `DivA = 8.8339`
  - `DivRe = 20.161`
  - `DivPr = 12.654`

### 2. Reranker appris pairwise sans SAM2

Nom court:
- `pairwise-temporal-v1`

Principe:
- meme generation de candidates
- apprentissage d'un score pairwise `bud, mother_candidate`
- application du score puis ILP global

Input:
- features heuristiques
- features spatio-temporelles agregees:
  - `temporal_support`
  - `contact_persistence`
  - `neck_persistence`
  - `attachment_persistence`
  - `dist_stability`
  - `separation_trend`
  - `framewise_best_fraction`

Training:
- supervise
- objectif `pairwise`
- modele lineaire/logistique sur features tabulaires

Forces:
- plus souple que l'heuristique pure
- exploite un signal temporel simple

Limites:
- pas de vrai contexte entre candidates
- pas d'information visuelle
- tendance a sur-apprendre plus vite que la baseline heuristique

Run:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1447_854f313_pairwise-temporal-v1`

Scores:
- `train_dev` F1: `0.697168`
- `val_holdout` F1: `0.641026`

### 3. Reranker appris pairwise avec SAM2

Nom court:
- `pairwise-temporal-sam2-v1`

Principe:
- meme principe que le pairwise
- ajout de similarities d'embeddings SAM2 par objet

Input:
- tout ce du pairwise sans SAM2
- plus embeddings SAM2 moyens sur masque d'objet:
  - `sam2_cosine_mean`
  - `sam2_cosine_min`
  - `sam2_cosine_trend`

Training:
- supervise
- objectif `pairwise`

Forces:
- ajoute un signal non purement geometrique
- ameliore le parentage pur

Limites:
- l'embedding est global a l'objet, pas localise au point d'emergence
- encore pas de vraie competition explicite entre candidates

Run:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1449_854f313_pairwise-temporal-sam2-v1`

Scores:
- `train_dev` F1: `0.700000`
- `val_holdout` F1: `0.653846`
- `val_holdout` diagnostic:
  - `stable_correct_parent = 40`
  - `stable_wrong_parent = 9`
  - `stable_orphan = 3`

### 4. Meilleur modele actuel: pairwise + SAM2 + blend heuristique

Nom court:
- `pairwise-temporal-sam2-blend060-v1`

Principe:
- score appris pairwise
- blend avec le score heuristique existant:
  - `score_final = 0.6 * score_appris + 0.4 * score_heuristique`
- puis ILP global

Input:
- identique au pairwise + SAM2

Training:
- supervise
- objectif `pairwise`
- score appris ensuite melange a l'heuristique

Forces:
- meilleur compromis actuel
- combine biais fort heuristique + signal appris
- le plus robuste parmi les modeles appris testes

Limites:
- reste un scoring par paire
- ne compare pas vraiment les candidates ensemble
- la limite restante est surtout dans les conflits locaux tres denses

Run:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1455_ce88ca0_pairwise-temporal-sam2-blend060-v1`

Scores:
- `train_dev` F1: `0.700000`
- `val_holdout` F1: `0.662420`
- `val_holdout` diagnostic:
  - `stable_correct_parent = 41`
  - `stable_wrong_parent = 9`
  - `stable_orphan = 2`
- Cell-HOTA:
  - `HOTA = 42.493`
  - `DivA = 10.276`
  - `DivRe = 23.283`
  - `DivPr = 14.46`

Interpretation:
- meilleur `DivA` observe
- meilleur parentage bud-centric observe
- meilleure reference actuelle

### 5. Transformer listwise sur features agregees

Nom court:
- `transformer-listwise-v1-postbase-dev-holdout`
- `transformer-listwise-sam2-v1-postbase-dev-holdout`

Principe:
- pour un bud, prendre toutes les candidates
- petit transformer listwise
- score par candidate
- ILP global ensuite

Input:
- features agregees par candidate
- version avec ou sans embeddings SAM2

Training:
- supervise
- loss listwise

Forces:
- formulation plus proche du ranking multi-candidates

Limites:
- l'entree reste encore trop tabulaire et agregee
- forte tendance au surapprentissage
- ne bat pas le pairwise + SAM2

Runs:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1632_1638d27_transformer-listwise-v1-postbase-dev-holdout`
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1649_1638d27_transformer-listwise-sam2-v1-postbase-dev-holdout`

Scores:
- sans SAM2:
  - `val_holdout` F1: `0.615385`
- avec SAM2:
  - `val_holdout` F1: `0.619355`

Conclusion:
- formulation pas encore rentable sur ce dataset

### 6. Context ranker sur dataset framewise + relatif

Nom court:
- `context-ranker-v2-dev-holdout`

Principe:
- construire un dataset contextuel par bud:
  - candidates
  - plusieurs frames
  - features relatives entre candidates
- petit modele appris
- ILP global ensuite

Input:
- tenseur `candidates x frames x features`
- features framewise:
  - `dist_norm`
  - `contact`
  - `neck`
  - `size_ratio`
  - `mother_age`
  - `dir_x`, `dir_y`
  - rang relatif par distance/contact
  - marges au meilleur
  - nombre de candidates vivantes
- features globales bud
- features globales candidate

Training:
- supervise
- cross-entropy listwise sur la bonne candidate

Forces:
- plus proche du vrai probleme
- integre enfin un contexte `framewise + relatif`
- bon support d'experimentation futur

Limites:
- la premiere version du dataset contextuel avait de la fuite GT
- version propre `v2` corrigee ensuite
- meme la version propre ne bat pas `pairwise + SAM2`

Run propre:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1721_0488f12_context-ranker-v2-dev-holdout`

Scores:
- `train_dev` F1: `0.734783`
- `val_holdout` F1: `0.636943`
- `val_holdout` diagnostic:
  - `stable_correct_parent = 37`
  - `stable_wrong_parent = 13`
  - `stable_orphan = 2`

Interpretation:
- mieux que la baseline heuristique sur le parentage brut
- moins bon que `pairwise + SAM2`

### 7. Context ranker regularise

Nom court:
- `context-ranker-v3-reg-dev-holdout`

Principe:
- meme dataset contextuel propre
- split interne de validation sur `train_dev`
- early stopping
- poids plus reguliers

Input:
- identique au context ranker

Training:
- supervise
- validation interne stricte par video

Forces:
- protocole plus propre
- surapprentissage reduit

Limites:
- baisse de performance holdout
- le gain de generalisation ne compense pas la perte de capacite utile

Run:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1803_3b875ba_context-ranker-v3-reg-dev-holdout`

Scores:
- `val_holdout` F1: `0.624204`

### 8. Context ranker regularise + interaction legere entre candidates

Nom court:
- `context-ranker-v4-attn-dev-holdout`

Principe:
- meme modele contextuel
- ajout d'une self-attention legere entre candidates

Input:
- identique au context ranker

Training:
- supervise
- meme protocole strict que `v3`

Forces:
- teste explicitement une competition legere entre candidates

Limites:
- n'apporte rien dans cette premiere version
- meme score que `v3`

Run:
- `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1804_3b875ba_context-ranker-v4-attn-dev-holdout`

Scores:
- `val_holdout` F1: `0.624204`

## Lecture Des Erreurs

Motif dominant restant:
- erreurs locales entre candidates tres proches
- surtout:
  - `ancestor_of_true`
  - `descendant_of_true`
  - parfois `sibling_of_true`

Ce que cela veut dire:
- le probleme n'est pas de choisir entre familles eloignees
- le vrai verrou est un ranking local dans une structure dense
- le modele choisit souvent le bon voisinage, mais le mauvais niveau local

Comparaison des `wrong_parent` restants sur `val_holdout`:
- baseline heuristique:
  - surtout `descendant_of_true` et `ancestor_of_true`
- `pairwise + SAM2`:
  - `ancestor_of_true = 4`
  - `descendant_of_true = 3`
  - `sibling_of_true = 1`
  - `other_branch = 1`
- context ranker v2:
  - `ancestor_of_true = 5`
  - `descendant_of_true = 5`
  - `sibling_of_true = 2`
  - `other_branch = 1`

Interpretation:
- `pairwise + SAM2` reste le meilleur pour resoudre ces conflits locaux

## Caveats Methodologiques

Important:
- les premiers essais contextuels avaient une fuite de label dans les features d'entree
- les runs a retenir pour ce modele sont ceux apres correction:
  - `context-ranker-v2`
  - `context-ranker-v3-reg`
  - `context-ranker-v4-attn`

Autre point:
- certaines erreurs metrico-evaluees sont discutees par l'expert:
  - petits buds detectes un peu plus tard
  - hallucinations de buds tres discutables
  - cas ambigus meme a l'oeil humain

Donc:
- `DivA` et `division_iou_window` restent utiles
- mais ils ne capturent pas parfaitement la qualite biologique percue

## Ressources Utiles

Scripts:
- generation dataset contextuel:
  - `/home/charvin-admin/Documents/cellSAM2/CellSam2/tools/build_bud_context_dataset.py`
- ranker contextuel:
  - `/home/charvin-admin/Documents/cellSAM2/CellSam2/tools/train_bud_context_ranker.py`
- reranker pairwise / transformer:
  - `/home/charvin-admin/Documents/cellSAM2/CellSam2/tools/learned_bud_rerank.py`
- rendu video GT vs prediction:
  - `/home/charvin-admin/Documents/cellSAM2/CellSam2/tools/render_parentage_comparison.py`

Videos de comparaison utiles:
- meilleur `pairwise + SAM2`:
  - `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1455_ce88ca0_pairwise-temporal-sam2-blend060-v1/review/video13_gt_vs_pairwise_sam2.mp4`
- meilleur contextuel:
  - `/home/charvin-admin/Documents/cellSAM2/experiments/20260312_1721_0488f12_context-ranker-v2-dev-holdout/review/video13_gt_vs_context_v2.mp4`

## Recommandation Courante

Pour un agent futur:
- prendre `pairwise-temporal-sam2-blend060-v1` comme meilleure reference
- ne pas pousser plus loin les heuristiques pures
- ne pas pousser plus loin le transformer listwise actuel
- si apprentissage plus riche:
  - partir du diagnostic local `ancetre / mere / descendante`
  - pas d'un contexte generique seulement
- si objectif applicatif immediat:
  - considerer que le systeme est deja bon
  - completer plutot l'evaluation ciblee `bas de cavite` et la revue expert

