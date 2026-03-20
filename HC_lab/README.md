HC_lab
dernier développement sur 

- check des données gt : masques, mantrack, cohérence des deux , 
- correction si erreur détectées (cellules sans parents, trop petites taille max de cellules)
- calcule la lignée cellulaire en post traitement de la segmentation et du tracking 
- evaluation de la détection des lignées cellulaire (détect, filiation, localistation temps et espaces), voir notebook compare_lineage_predictor.ipynb

-> Clean effectué sur les dernières données moma_N_0_rechecked_cleaned.zip (sur le slack)

Editer chemin et paramètres directement dans les fichiers


pour nettoyer les artefacts de masque dans la gt
`python HC_lab/clean_data.py` 

 pour obtenir une video de la gt

`python HC_lab/visu_tools.py` 

prédiction avec cellsam2 avec évaluation des divisions
`python inference/track_cells.py --video_path /home/hcourtei/Projects/Cell_proj/CellSam2Gilles/HC_lab/moma_N_0_rechecked_cleaned/val/CTC   --model_name moma_N0_rechecked_v100   --res_path eval_model  --data_version moma_N_0_rechecked_cleaned`

- prédiction avec post_process des tracklets, voir notebook

https://github.com/gcharvin/CellSam2/blob/upgrade_div2/HC_lab/compare_lineage_predictor.ipynb