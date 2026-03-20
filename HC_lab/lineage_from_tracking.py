"""
Module pour le suivi de lignée cellulaire à partir de masques de segmentation.

Ce module fournit une classe `LineageFromTracking` qui permet de :
1. Charger et analyser les masques de segmentation cellulaire
2. Calculer les durées de vie des cellules
3. Établir les relations de lignée parent-enfant
4. Sauvegarder les résultats dans un format compatible avec les outils d'évaluation

Fonctionnalités principales :
- Détection des centroïdes cellulaires avec OpenCV
- Filtrage des cellules par durée de vie minimale
- Appariement des cellules entre frames consécutives
- Calcul des relations de lignée basées sur la proximité spatiale
- Sauvegarde des résultats au format lineage_from_track.txt

"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CellInfo:
    """Stocke les informations d'une cellule"""
    cell_id: int
    begin_frame: int
    end_frame: int
    centroid: tuple  # (x, y)
    area: int

class LineageFromTracking:
    def __init__(self, mask_dir: Path, prefix: str = 'mask'):
        self.mask_dir = mask_dir
        self.prefix = prefix
        self.cell_infos: Dict[int, CellInfo] = {}
        self.mask_files: List[str] = []

        # Initialiser les fichiers masques
        self._init_mask_files()

    def _init_mask_files(self):
        """Initialise la liste des fichiers masques triés"""
        self.mask_files = sorted([
            f for f in os.listdir(self.mask_dir)
            if f.startswith(self.prefix) and f.endswith(".tif")
        ])

    def load_cell_lifetimes(self, min_lifetime: int = 5) -> pd.DataFrame:
        """
        Charge les masques et construit le tableau de durées de vie
        Args:
            min_lifetime: Durée de vie minimale pour conserver une cellule
        Returns:
            DataFrame avec cell_id, begin_frame, end_frame, life_time
        """
        # Charger les IDs de cellules par frame
        frame_cell_ids = self._load_cell_ids_per_frame()

        # Construire le dictionnaire cell_frames
        cell_frames = {}
        for frame_idx, ids in frame_cell_ids.items():
            for cid in ids:
                cell_frames.setdefault(cid, []).append(frame_idx)

        # Construire le DataFrame
        records = []
        for cid, frames in cell_frames.items():
            frames = sorted(frames)
            diffs = [j - i for i, j in zip(frames[:-1], frames[1:])]

            if any(d != 1 for d in diffs):
                print(f"Warning: cell {cid} non contiguë -> frames {frames}")

            # Calculer le centroïde de la cellule sur toute sa durée de vie
            centroid = self._compute_cell_centroid(cid, frames)

            records.append({
                "cell_id": cid,
                "begin_frame": frames[0],
                "end_frame": frames[-1],
                "life_time": len(frames),
                "centroid": centroid,
                "area": len(frames)  # Placeholder, à calculer proprement
            })

        df = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)
        df["life_time"] = df["end_frame"] - df["begin_frame"] + 1

        # Filtrer par durée de vie minimale
        df_filtered = df[df["life_time"] >= min_lifetime].copy()
        df_filtered.reset_index(drop=True, inplace=True)

        return df_filtered

    def _load_cell_ids_per_frame(self) -> Dict[int, List[int]]:
        """Charge les IDs de cellules pour chaque frame"""
        frame_cell_ids = {}

        for frame_idx, fname in enumerate(self.mask_files):
            path = os.path.join(self.mask_dir, fname)
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            cell_ids = np.unique(mask)
            cell_ids = cell_ids[cell_ids != 0]  # Exclure le fond
            frame_cell_ids[frame_idx] = cell_ids.tolist()

        return frame_cell_ids

    def _compute_cell_centroid(self, cell_id: int, frames: List[int]) -> tuple:
        """Calcule le centroïde moyen d'une cellule sur plusieurs frames"""
        centroids = []

        for frame_idx in frames:
            path = os.path.join(self.mask_dir, self.mask_files[frame_idx])
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            cell_mask = (mask == cell_id).astype(np.uint8)
            moments = cv2.moments(cell_mask)

            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                centroids.append((cx, cy))

        if centroids:
            avg_x = int(np.mean([c[0] for c in centroids]))
            avg_y = int(np.mean([c[1] for c in centroids]))
            return (avg_x, avg_y)
        return (-1, -1)

    def compute_lineage(self, df_cells: pd.DataFrame,
                        max_distance: int = 40,
                        min_parent_age: int = 1,
                        enforce_parent_larger: bool = True) -> pd.DataFrame:
        """
        Calcule les relations de lignée entre cellules
        Args:
            df_cells: DataFrame avec les durées de vie des cellules
            max_distance: Distance maximale pour considérer comme même cellule
            min_parent_age: Âge minimal du parent
            enforce_parent_larger: Si True, le parent doit être plus grand
        Returns:
            DataFrame avec cell_id, begin_frame, end_frame, parent_id
        """
        # Créer un index pour accès rapide
        df_index = df_cells.set_index("cell_id")
        valid_ids = set(df_cells["cell_id"])

        parent_ids = []

        for _, row in df_cells.iterrows():
            cid = row["cell_id"]
            begin_frame = row["begin_frame"]

            # Cellules initiales (frame 0)
            if begin_frame == 0:
                parent_ids.append(0)
                continue

            # Charger le masque de la frame de début
            path = os.path.join(self.mask_dir, self.mask_files[begin_frame])
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            # Cellule fille
            binary_child = (mask == cid).astype(np.uint8)
            if binary_child.sum() == 0:
                parent_ids.append(-1)
                continue

            # Centroïde et aire de la cellule fille
            moments_child = cv2.moments(binary_child)
            if moments_child["m00"] != 0:
                child_centroid = (int(moments_child["m10"] / moments_child["m00"]),
                                 int(moments_child["m01"] / moments_child["m00"]))
            else:
                parent_ids.append(-1)
                continue

            child_area = binary_child.sum()

            # Trouver les candidats parents
            candidates = []
            candidate_distances = []

            for other_id in valid_ids:
                if other_id == cid: continue

                other_row = df_index.loc[other_id]
                # Parent doit être présent à cette frame
                if not (other_row["begin_frame"] <= begin_frame <= other_row["end_frame"]):
                    continue

                # Âge minimal du parent
                age = begin_frame - other_row["begin_frame"]
                if age < min_parent_age:
                    continue

                # Masque parent
                binary_parent = (mask == other_id).astype(np.uint8)
                if binary_parent.sum() == 0:
                    continue

                parent_area = binary_parent.sum()

                # Option : parent doit être plus grand
                if enforce_parent_larger and parent_area < child_area:
                    continue

                # Centroïde parent
                moments_parent = cv2.moments(binary_parent)
                if moments_parent["m00"] != 0:
                    parent_centroid = (int(moments_parent["m10"] / moments_parent["m00"]),
                                     int(moments_parent["m01"] / moments_parent["m00"]))
                else:
                    continue

                # Distance entre parent et enfant
                dist = np.linalg.norm(np.array(child_centroid) - np.array(parent_centroid))
                if dist > max_distance:
                    continue

                candidates.append(other_id)
                candidate_distances.append(dist)

            # Choix du meilleur parent
            if not candidates:
                parent_ids.append(-1)
                continue

            best_idx = np.argmin(candidate_distances)
            parent_ids.append(candidates[best_idx])

        # Ajouter la colonne parent_id
        df_result = df_cells.copy()
        df_result["parent_id"] = parent_ids
        self.df_lineage = df_result

        return df_result

    def save_lineage(self, output_path: Path, header: bool = False):
        """
        Sauvegarde le DataFrame lineage dans un fichier lineage_from_track.txt

        Args:
            output_path: Chemin vers le fichier de sortie
            header: Si True, écrit les en-têtes de colonnes (défaut: False)
        """
        if self.df_lineage is None:
            raise ValueError("Aucun DataFrame de tracking disponible. Exécutez compute_lineage() d'abord.")

        # Créer le répertoire de sortie s'il n'existe pas
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_to_save = self.df_lineage[['cell_id', 'begin_frame', 'end_frame', 'parent_id']]
        # Sauvegarder avec pandas en utilisant un espace comme séparateur
        df_to_save.to_csv(
            output_path,
            sep=' ',  # Utilise un espace comme séparateur
            header=header,
            index=False
        )

        print(f"Lineage saved to {output_path}")



if __name__ == "__main__":
    # Configuration
    mask_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/eval_model/model:moma_N3_checked_v100/data_vers:moma_N3_checked/12")
    min_lifetime = 5
    # 1. Initialiser l'analyser
    lineage_analyzer = LineageFromTracking(mask_dir)
    # 2. Charger les durées de vie
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

    # Comparaison avec GT
    # gt_cleaned_mask_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/HC_lab/moma_N_0_rechecked_cleaned/val/CTC/12_GT/TRA")
    # man_track_df = pd.read_csv(gt_cleaned_mask_dir / "man_track.txt", sep='\s+', header=None,
    #                           names=['cell_id', 'begin_frame', 'end_frame', 'parent_id'])
    # man_track_df.insert(3, "life_time", man_track_df["end_frame"] - man_track_df["begin_frame"] + 1)
    #
    # print("\nMAN_TRACK")
    # print(man_track_df.to_markdown(index=False))
    # print(f"Nb cellules lignées GT: {len(man_track_df)}")
