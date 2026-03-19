from pathlib import Path
import json
from HC_lab.lineage_from_tracking import LineageFromTracking
from HC_lab.metrics_division import load_full_division_events, evaluate_single_prediction

# Configuration

temporal_tolerance = 3
max_future_frame_offset = 2
iou_thresh_mother = 0.5
iou_thresh_bud = 0.2
pred_mask_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/eval_model/model:moma_N3_checked_v100/data_vers:moma_N3_checked/12")
gt_mask_dir = Path("/home/hcourtei/Projects/Cell_proj/data/moma_N_3_checked/moma/val/CTC/12_GT/TRA")

# 1. Initialiser l'analyser
lineage_analyzer = LineageFromTracking(pred_mask_dir, prefix = 'mask')

# 2. Charger les durées de vie
min_lifetime = 5
df_lifetimes = lineage_analyzer.load_cell_lifetimes(min_lifetime=min_lifetime)
print(f"{len(df_lifetimes)} cellules conservées après filtrage")

# 3. Calculer les relations de lignée
df_lineage = lineage_analyzer.compute_lineage(
    df_lifetimes,
    max_distance=35,
    min_parent_age=1,
    enforce_parent_larger=True
)

# 4. Afficher les résultats
print("\nPRED_TRACK")
print(df_lineage.to_markdown(index=False))
print(f"Nb cellules lignées prédites: {len(df_lineage)}")

# 5. Sauvegarder le résultat
lineage_from_track_path = Path("./lineage_from_track.txt")
lineage_analyzer.save_lineage(lineage_from_track_path)
print(f"Résultat sauvegardé dans {lineage_from_track_path}")

# =============================================
# ÉVALUATION DES DIVISIONS
# =============================================
print("\n" + "="*60)
print("ÉVALUATION DES DIVISIONS")
print("="*60)

# Charger les événements GT
gt_events = load_full_division_events(pred_mask_dir / "summary" / "man_track.txt")
# Charger les événements prédits par celssam2
pred_events = load_full_division_events(pred_mask_dir / "summary" / "res_track.txt")

print("lineage from model cellsam2")
results = evaluate_single_prediction(
    gt_mask_dir=gt_mask_dir,
    man_track_path=pred_mask_dir / "summary" / "man_track.txt",
    pred_mask_dir=pred_mask_dir,
    pred_lineage_path=pred_mask_dir / "summary" / "res_track.txt",
    temporal_tolerance=3,
    max_future_frame_offset=2,
    iou_thresh_mother=0.5,
    iou_thresh_bud=0.2
)

print("Résultats de l'évaluation:")
print(json.dumps(results, indent=2))
print("="*60)
print("lineage from model lineage_from_track_path")
results2 = evaluate_single_prediction(
    gt_mask_dir=gt_mask_dir,
    man_track_path=pred_mask_dir / "summary" / "man_track.txt",
    pred_mask_dir=pred_mask_dir,
    pred_lineage_path=lineage_from_track_path,
    temporal_tolerance=3,
    max_future_frame_offset=2,
    iou_thresh_mother=0.5,
    iou_thresh_bud=0.2
)

print("Résultats de l'évaluation:")
print(json.dumps(results2, indent=2))
