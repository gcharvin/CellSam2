
check des données gt : masques, mantrack, cohérence des deux , correction si erreur détectées (cellules sans parents, trop petites taille max de cellules)

Editer chemin et paramètres directement dans les fichiers

`python HC_lab/clean_data.py` pour nettoyer une vidéo

`python HC_lab/visu_tools.py`  pour obtenir une video de la gt



# DIVERS avant
# PNRIA
 python train_ctc.py   launcher.experiment_log_dir=sam2_logs/CellSam2-tracking   scratch.dataset_name=moma  dataset.data_dir=/home/hcourtei/Projects/Cell_proj/data/moma_N_3_checked/moma
données avec erreurs
# STRUCTURE
preprocess:
    detect anomalies dans gt : mask and id (too short lived, min area on max lifetime, parentless)
    correct mask and save with same structure as datadir
    update man_track.txt
visu_tools
    gt : img + cell + opt id + div from mantrack
    pred vs gt : segment + tracking + div from tracking  + gt
eval :
   metrics : division (détect, filiation, localistation temps et espaces)

infer from cellsam2 : infer + visu pred vs gt + metrics

CTC format
https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf


python3 tools/eval_division_iou_window.py   --gt-root /home/hcourtei/Projects/Cell_proj/data/moma_N_1_checked/moma/val/CTC/ 
                                            --pred-root /home/hcourtei/Projects/Cell_proj/CellSam2/results/tracking/    
                                            --videos "12,13,14"
                                            --window 3
                                            --iou-thresh 0.5
                                            --gt-mask-prefix "man_track"


                        --save-json /home/hcourtei/Projects/Cell_proj/CellSam2/results/evaluation_results.json
python3 inference/track_cells.py    --video_path /home/hcourtei/Projects/Cell_proj/data/moma_N_1_checked/moma/val/CTC/12 
                                    --model_name moma_N3_checked_v100  
                                    --res_path results/moma_N3_checked_v100


# merge videos

 export PYTHONPATH="/home/hcourtei/Projects/Cell_proj/CellSam2Gilles:$PYTHONPATH"

```
ffmpeg \
  -i pred_track_moma_N0_checked_v100_video_13.mp4 \
  -i pred_track_moma_N1_checked_v100_video_13.mp4 \
  -i pred_track_moma_N3_checked_v100_video_13.mp4 \
  -i pred_track_moma_N5_checked_v100_video_13.mp4 \
  -i pred_track_moma_N7_checked_v100_video_13.mp4 \
  -i pred_track_moma_N9_checked_v100_video_13.mp4 \
  -i 13_overlay.mp4 \
  -f lavfi -i color=c=black:s=66x136:d=16 \
  -filter_complex "\
[0:v]scale=66:136,drawtext=text='N0':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v0]; \
[1:v]scale=66:136,drawtext=text='N1':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v1]; \
[2:v]scale=66:136,drawtext=text='N3':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v2]; \
[6:v]scale=66:136,drawtext=text='N3 GT':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v6]; \
[3:v]scale=66:136,drawtext=text='N5':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v3]; \
[4:v]scale=66:136,drawtext=text='N7':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v4]; \
[5:v]scale=66:136,drawtext=text='N9':x=main_w-tw-2:y=2:fontsize=10:fontcolor=white[v5]; \
[7:v]scale=66:136[empty]; \
[v0][v1][v2][v6]hstack=inputs=4[top]; \
[v3][v4][v5][empty]hstack=inputs=4[bottom]; \
[top][bottom]vstack=inputs=2[out]" \
-map "[out]" -c:v libx264 -crf 18 -preset fast \
grid_moma_checked_v100_video_13_labeled_topright_overlay_fixed.mp4
```


CellSAM2 is an