"""
Script de validation, analyse et nettoyage des données de suivi cellulaire (cell tracking).
Ce script lit des masques d'images et des fichiers de suivi manuel (man_track.txt),
vérifie leur cohérence, extrait les informations cellulaires, et nettoie les données si nécessaire.

Fonctionnalités principales :
- Validation des IDs, frames, relations parent-enfant, durée de vie et aire des cellules.
- Extraction des données des masques (aires, centroïdes, frames d'apparition).
- Nettoyage des données : suppression des cellules problématiques, recalcul des IDs, sauvegarde des données corrigées.
- Automatisation du traitement pour plusieurs vidéos (train/val).
"""
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CellInfo:
    """Structure pour stocker les informations d'une cellule"""
    cell_id: int
    begin_frame: int
    end_frame: int
    max_area: int
    living_time: int
    frames: List[int]
    areas: List[int]
    centroids: List[Tuple[int, int]]

class MaskLineageCellGT:
    def __init__(self, data_dir: Path, id_video: int, min_live_cell=2, min_area_cell=30):
        self.id_video = id_video
        self.data_dir = data_dir
        self.mask_dir = data_dir / f"{id_video:02d}_GT/TRA"
        self.man_track_path = self.mask_dir / 'man_track.txt'
        self.track_df = pd.read_csv(self.man_track_path, sep='\s+', header=None,
                               names=['cell_id', 'begin_frame', 'end_frame', 'parent_id'])

        self.min_live_cell = min_live_cell  # temps minimum de vie d'une cellule
        self.min_area_cell =min_area_cell # minimum pour le maximum des aires de la cellule trackée
        self.cell_data = {}
        self.cell_data_from_mask = {}
        self.cell_ids_to_correct = set()  # un set



    def _check_ids_continuous(self) -> bool:
        """Vérifie que les IDs sont numérotés de 1 à N sans trous"""
        cell_ids = sorted(self.track_df['cell_id'].unique())
        expected_ids = list(range(1, len(cell_ids) + 1))
        if cell_ids != expected_ids:
            print(f"❌ IDs invalides: attendu {expected_ids}, obtenu {cell_ids}")
            self.cell_ids_to_correct.update(cell_ids)
            return False
        return True

    def _check_frames_valid(self) -> bool:
        """Vérifie que begin_frame < end_frame pour chaque cellule"""
        invalid_frames = self.track_df[self.track_df['begin_frame'] >= self.track_df['end_frame']]
        if not invalid_frames.empty:
            cell_ids = invalid_frames['cell_id'].tolist()
            print(f"❌ Frames invalides: cellules {cell_ids}")
            self.cell_ids_to_correct.update(cell_ids)
            return False
        return True

    def _check_parentless_cells(self) -> bool:
        """Vérifie que parent_id=0 seulement pour les cellules à la frame 0"""
        min_frame = self.track_df['begin_frame'].min()
        parentless_not_at_start = self.track_df[
            (self.track_df['parent_id'] == 0) &
            (self.track_df['begin_frame'] > min_frame)
            ]
        if not parentless_not_at_start.empty:
            cell_ids = parentless_not_at_start['cell_id'].tolist()
            print(f"❌ Cellules sans parent non à la frame 0: {cell_ids}")
            self.cell_ids_to_correct.update(cell_ids)
            return False
        return True

    def _check_parent_child_relationships(self) -> bool:
        """Vérifie que la cellule fille naît pendant la vie du parent"""
        # D'abord créer un dictionnaire des informations parent
        parent_info = {}
        for _, row in self.track_df.iterrows():
            cell_id = row['cell_id']
            parent_info[cell_id] = {
                'begin_frame': row['begin_frame'],
                'end_frame': row['end_frame']
            }

        val = True
        # Ensuite vérifier chaque cellule avec un parent
        for _, row in self.track_df.iterrows():
            parent_id = row['parent_id']
            if parent_id != 0:
                if parent_id not in parent_info:
                    print(f"❌ Parent {parent_id} introuvable pour cellule {row['cell_id']}")
                    self.cell_ids_to_correct.add(row['cell_id'])
                    val = False
                    continue

                parent_begin = parent_info[parent_id]['begin_frame']
                parent_end = parent_info[parent_id]['end_frame']
                child_begin = row['begin_frame']

                # Vérifier UNIQUEMENT que la cellule fille naît pendant la vie du parent
                if not (parent_begin <= child_begin <= parent_end):
                    print(f"❌ Cellule {row['cell_id']} (begin={child_begin}) "
                          f"ne naît pas pendant la vie du parent {parent_id} "
                          f"(begin={parent_begin}, end={parent_end})")
                    self.cell_ids_to_correct.add(row['cell_id'])
                    val = False
        return val

    def _check_lifetime(self) -> bool:
        """Vérifie que les cellules ont une taille et une durée de vie minimales"""

        # Vérifier la durée de vie minimale
        short_lived_cells = self.track_df[
            self.track_df['end_frame'] - self.track_df['begin_frame'] + 1 < self.min_live_cell]
        if not short_lived_cells.empty:
            cell_ids = short_lived_cells['cell_id'].tolist()
            print(f"❌ Cellules avec durée de vie trop courte (<{self.min_live_cell} frames): {cell_ids}")
            self.cell_ids_to_correct.update(cell_ids)

        return short_lived_cells.empty

    def check_man_track(self) -> bool:
        """Vérifie la validité du fichier man_track.txt"""
        print("-" * 60)
        print("-> verification man_track.txt")

        val_man_track = (self._check_ids_continuous()
                       & self._check_frames_valid()
                       & self._check_parentless_cells()
                       & self._check_parent_child_relationships()
                       & self._check_lifetime()
                         )


        return val_man_track

    def extract_mask_data(self) -> Dict[int, CellInfo]:
        """Extrait les données des masques et reconstruit les informations cellulaires"""
        cell_data_from_mask = {}
        print("-" * 60)
        print("-> verification masques")
        for mask_file in sorted(self.mask_dir.glob("man_track*.tif")):
            try:
                frame_idx = int(mask_file.stem.replace("man_track", ""))
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

                if mask is None:
                    print(f"⚠️ Impossible de charger le masque {mask_file}")
                    continue

                unique_ids = np.unique(mask)
                unique_ids = unique_ids[unique_ids != 0]

                for cell_id in unique_ids:
                    cell_id = int(cell_id)
                    cell_mask = (mask == cell_id).astype(np.uint8)
                    area = int(np.sum(cell_mask))

                    # Calculer le centroïde
                    moments = cv2.moments(cell_mask)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                    else:
                        cx, cy = -1, -1

                    # Mettre à jour ou créer l'entrée CellInfo
                    if cell_id in cell_data_from_mask:
                        cell_data_from_mask[cell_id].frames.append(frame_idx)
                        cell_data_from_mask[cell_id].areas.append(area)
                        cell_data_from_mask[cell_id].centroids.append((cx, cy))
                        cell_data_from_mask[cell_id].begin_frame = min(cell_data_from_mask[cell_id].begin_frame, frame_idx)
                        cell_data_from_mask[cell_id].end_frame = max(cell_data_from_mask[cell_id].end_frame, frame_idx)
                        cell_data_from_mask[cell_id].max_area = max(cell_data_from_mask[cell_id].max_area, area)
                    else:
                        cell_data_from_mask[cell_id] = CellInfo(
                            cell_id=cell_id,
                            begin_frame=frame_idx,
                            end_frame=frame_idx,
                            max_area=area,
                            living_time=0,
                            frames=[frame_idx],
                            areas=[area],
                            centroids=[(cx, cy)]
                        )


            except Exception as e:
                print(f"⚠️ Erreur lors du traitement du masque {mask_file}: {e}")


        # Calculer le living_time pour chaque cellule
        for cell_id in cell_data_from_mask:
            cell_data_from_mask[cell_id].living_time = len(cell_data_from_mask[cell_id].frames)

        small_cells = []
        for cell_id, cell_info in cell_data_from_mask.items():
            # Vérifier la taille minimale
            if cell_info.max_area < self.min_area_cell:
                small_cells.append(cell_id)

        if small_cells:
            print(f"❌ Cellules avec aire maximale trop petite (<{self.min_area_cell} pixels) :  {small_cells}")
            self.cell_ids_to_correct.update(small_cells)

        self.cell_data_from_mask =cell_data_from_mask

        return cell_data_from_mask

    def check_data(self)-> bool:
        """Vérifie la cohérence complète des données"""
        print(f"Vérification des données dans {self.mask_dir}")

        # 1. Vérification du fichier man_track
        val_man_track = self.check_man_track()
        if val_man_track:
            print("✅ Fichier man_track valide")

        global_val = val_man_track
        # 2. Extraction des données des masques
        mask_data = self.extract_mask_data()
        # pprint(mask_data)
        # 3. Vérification de la cohérence entre masques et tracking
        print("-" * 60)
        print("-> vérification cohérence")

        # Comparer les IDs
        mask_ids = set(self.cell_data_from_mask.keys())
        track_ids = set(self.track_df['cell_id'].unique())

        missing_in_track = mask_ids - track_ids
        if missing_in_track:
            print(f"❌ Cellules dans masques mais pas dans tracking: {list(missing_in_track)}")
            self.cell_ids_to_correct.update(missing_in_track)

        missing_in_masks = track_ids - mask_ids
        if missing_in_masks:
            print(f"❌ Cellules dans tracking mais pas dans masques: {list(map(int, missing_in_masks))}")
            self.cell_ids_to_correct.update(missing_in_masks)

        # Vérifier la cohérence des frames

        for cell_id, cell_info in self.cell_data_from_mask.items():
            if cell_id in track_ids:
                track_info = self.track_df[self.track_df['cell_id'] == cell_id].iloc[0]

                if (cell_info.begin_frame != track_info['begin_frame'] or
                        cell_info.end_frame != track_info['end_frame']):
                    print(f"❌ Incohérence de frames pour cellule {cell_id}: "
                          f"masques {cell_info.begin_frame}-{cell_info.end_frame} vs "
                          f"tracking {track_info['begin_frame']}-{track_info['end_frame']}")
                    self.cell_ids_to_correct.add(cell_id)
                    global_val &= False
        return global_val

    def clean_data(self, out_data_dir: Path):
        """Nettoie les données en appliquant simplement le mapping des IDs"""
        print("Début du nettoyage des données...")

        # Créer les répertoires de sortie
        out_mask_dir = out_data_dir / f"{self.id_video:02d}_GT/TRA"
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        # Créer le répertoire pour les images
        out_img_dir = out_data_dir / f"{self.id_video:02d}"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        img_dir = self.data_dir / f"{self.id_video:02d}"

        # 1. Copier les images originales
        print("Copie des images originales...")
        for img_file in sorted(img_dir.glob("t*.tif")):
            dest_file = out_img_dir / img_file.name
            shutil.copy(img_file, dest_file)
        print(f"Images copiées dans {out_img_dir}")

        # 2. Nettoyer le fichier man_track
        print("Nettoyage du fichier man_track...")
        cleaned_track_df = self.track_df.copy()

        # Supprimer les cellules problématiques
        cleaned_track_df = cleaned_track_df[~cleaned_track_df['cell_id'].isin(self.cell_ids_to_correct)]

        # Créer le mapping old_id -> new_id
        old_ids = sorted(cleaned_track_df['cell_id'].unique())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}

        # Appliquer le mapping
        cleaned_track_df['cell_id'] = cleaned_track_df['cell_id'].map(id_mapping)
        cleaned_track_df['parent_id'] = cleaned_track_df['parent_id'].map(
            lambda x: id_mapping.get(x, 0) if x != 0 else 0
        )

        # Sauvegarder le fichier nettoyé
        out_man_track_path = out_mask_dir / 'man_track.txt'
        cleaned_track_df.to_csv(out_man_track_path, sep=' ', header=False, index=False)
        print(f"Fichier man_track nettoyé sauvegardé dans {out_man_track_path}")

        # 3. Nettoyer les masques
        print("Nettoyage des masques...")
        for mask_file in sorted(self.mask_dir.glob("man_track*.tif")):
            try:
                frame_idx = int(mask_file.stem.replace("man_track", ""))
                out_mask_file = out_mask_dir / f"man_track{frame_idx:03d}.tif"
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

                if mask is None:
                    print(f"⚠️ Impossible de charger le masque {mask_file}")
                    continue

                # Créer une copie du masque
                cleaned_mask = mask.copy()

                # Supprimer les cellules à corriger (remplacer par du fond noir = 0)
                for cell_id in self.cell_ids_to_correct:
                    cleaned_mask[mask == cell_id] = 0

                # Appliquer le mapping aux cellules restantes
                unique_ids = np.unique(cleaned_mask)
                unique_ids = unique_ids[unique_ids != 0]  # Exclure le fond (0)

                for cell_id in unique_ids:
                    cell_id = int(cell_id)
                    if cell_id in id_mapping:
                        new_id = id_mapping[cell_id]
                        cleaned_mask[cleaned_mask == cell_id] = new_id

                cv2.imwrite(str(out_mask_file), cleaned_mask)

            except Exception as e:
                print(f"⚠️ Erreur lors du nettoyage du masque {mask_file}: {e}")

        print("Nettoyage des données terminé avec succès!")
        print(f"Données nettoyées sauvegardées dans {out_data_dir}")

    def copy_data(self, out_data_dir: Path):
        """Copie directement les données sans modification"""
        print("Aucune cellule à corriger, copie directe des données...")

        # Créer les répertoires de sortie
        out_mask_dir = out_data_dir / f"{self.id_video:02d}_GT/TRA"
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        # Créer le répertoire pour les images
        out_img_dir = out_data_dir / f"{self.id_video:02d}"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        img_dir = self.data_dir / f"{self.id_video:02d}"

        # 1. Copier les images originales
        print("Copie des images originales...")
        for img_file in sorted(img_dir.glob("t*.tif")):
            dest_file = out_img_dir / img_file.name
            shutil.copy(img_file, dest_file)
        print(f"Images copiées dans {out_img_dir}")

        # 2. Copier le fichier man_track original
        print("Copie du fichier man_track...")
        shutil.copy(self.man_track_path, out_mask_dir / 'man_track.txt')
        print(f"Fichier man_track copié dans {out_mask_dir}")

        # 3. Copier les masques originaux
        print("Copie des masques...")
        for mask_file in sorted(self.mask_dir.glob("man_track*.tif")):
            dest_file = out_mask_dir / mask_file.name
            shutil.copy(mask_file, dest_file)
        print(f"Masques copiés dans {out_mask_dir}")

        print("Copie directe des données terminée!")


if __name__=='__main__':
    min_live_cell=2
    min_area_cell=30
    data_root_dir = Path("/home/hcourtei/Projects/Cell_proj/data/moma_N_0_rechecked/moma")
    output_root_dir = Path("./moma_N_0_rechecked_cleaned")
    for mode in ["train", "val"]:
        id_videos = [12,13,14] if mode == "val" else list(range(1,12))
        data_dir = data_root_dir/ mode / "CTC"
        output_dir = output_root_dir / mode / "CTC"
        for  id_video in id_videos:
            gt_analyser = MaskLineageCellGT(data_dir, id_video, min_live_cell, min_area_cell)
            if gt_analyser.check_data():
                print("✅ Cohérence vérifiée entre les masques et son man_track.txt")
                gt_analyser.copy_data(output_dir)
            else:
                print(f"❌ Cellules à corriger: {sorted(gt_analyser.cell_ids_to_correct)}")
                gt_analyser.clean_data(output_dir)

                # gt_analyser2 = MaskLineageCellGT(output_dir, id_video, min_live_cell, min_area_cell)
                # if gt_analyser2.check_data():
                #     print("✅ Cohérence vérifiée entre les masques et son man_track.txt")