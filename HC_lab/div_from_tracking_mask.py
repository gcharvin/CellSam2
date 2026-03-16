import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
from typing import Dict, List, Tuple, Optional



def link_new_cells_to_previous(centroids_by_frame) -> Dict[int, int]:
    """Établit les relations de lignée entre les cellules de frames consécutives.

    Args:
        centroids_by_frame: Dictionnaire des centroïdes par frame et par cellule.

    Returns:
        Dictionnaire des relations de lignée (cell_id: parent_id).
    """
    cell_lineage = {}
    sorted_frames = sorted(centroids_by_frame.keys())

    for i in range(1, len(sorted_frames)):
        current_frame = sorted_frames[i]
        previous_frame = sorted_frames[i-1]

        current_centroids = centroids_by_frame[current_frame]
        previous_centroids = centroids_by_frame[previous_frame]

        # Trouver les nouvelles cellules dans la frame actuelle
        new_cells = set(current_centroids.keys()) - set(previous_centroids.keys())

        for cell_id in new_cells:
            current_centroid = current_centroids[cell_id]
            min_distance = float('inf')
            closest_cell_id = None

            # Trouver la cellule la plus proche dans la frame précédente
            for prev_cell_id, prev_centroid in previous_centroids.items():
                dist = distance.euclidean(current_centroid, prev_centroid)
                if dist < min_distance:
                    min_distance = dist
                    closest_cell_id = prev_cell_id

            if closest_cell_id is not None:
                cell_lineage[cell_id] = closest_cell_id

    return cell_lineage

def extract_cell_ids_from_masks(mask_dir: Path) -> Dict[int, List[int]]:
    """Extrait les identifiants des cellules pour chaque frame.

    Args:
        mask_dir: Répertoire contenant les masques de segmentation.

    Returns:
        Dictionnaire des identifiants de cellules par frame.
    """
    cell_ids_by_frame = {}

    for mask_file in sorted(mask_dir.glob("mask*.tif")):
        frame_idx = int(mask_file.stem.replace("mask", ""))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]  # Exclure le fond (0)
        cell_ids_by_frame[frame_idx] = unique_ids.tolist()

    return cell_ids_by_frame

def build_tracking_table(cell_ids_by_frame: Dict[int, List[int]],
                         cell_lineage: Dict[int, int]) -> pd.DataFrame:
    """Constitue un tableau de suivi des cellules avec les informations de lignée.

    Args:
        cell_ids_by_frame: Dictionnaire des identifiants de cellules par frame.
        cell_lineage: Dictionnaire des relations de lignée.

    Returns:
        DataFrame de suivi des cellules avec les informations de lignée.
    """
    cell_tracking = {}

    for frame_idx, cell_ids in cell_ids_by_frame.items():
        for cell_id in cell_ids:
            if cell_id not in cell_tracking:
                cell_tracking[cell_id] = {
                    'begin_frame': frame_idx,
                    'end_frame': frame_idx,
                    'parent_id': 0 if frame_idx == min(cell_ids_by_frame.keys()) else cell_lineage.get(cell_id, np.nan)
                }
            else:
                cell_tracking[cell_id]['end_frame'] = frame_idx

    tracking_table = pd.DataFrame.from_dict(cell_tracking, orient='index')
    tracking_table.reset_index(inplace=True)
    tracking_table.rename(columns={'index': 'cell_id'}, inplace=True)

    return tracking_table
