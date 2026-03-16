import cv2
import numpy as np
from pathlib import Path
from HC_lab.utils import read_image, get_centroid


def make_gt_frame_color(image_path: Path,
                        mask_path: Path,
                        frame_idx: int,
                        fontscale: float = 0.3,
                        gamma_img: float = 0.7,
                        thickness: int = 1) -> np.ndarray:
    """Visualise une image en niveaux de gris avec les contours en rouge (version couleur)"""
    # Lire l'image en niveaux de gris
    red, green, white = (0, 0, 255), (0, 255, 0), (255, 255, 255)
    gray_image = read_image(str(image_path), gamma_img)

    # Charger le masque
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    # Créer une image couleur pour les contours (3 canaux)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Dessiner les contours en rouge
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # Exclure le fond (0)

    for cell_id in unique_ids:
        # Dessiner les contours
        cell_mask = (mask == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_image, contours, -1, red, thickness)  # Rouge

        # Dessiner les IDs au centroïde en vert
        cx, cy = get_centroid(mask, int(cell_id))
        if cx != -1 and cy != -1:
            cv2.putText(color_image,
                        str(cell_id),
                        (cx - 5, cy + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, fontscale, green, thickness, cv2.LINE_AA)  # Vert
            # Ajouter le numéro de frame en blanc en haut à gauche
        cv2.putText(color_image, f"{frame_idx}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1, cv2.LINE_AA)
    return color_image


def make_video_gt(img_dir: Path,
                   mask_dir: Path,
                   output_path: Path,
                   video_fps: int = 10,
                   gamma_img: float = 0.7,
                   fontscale: float = 0.3,
                   thickness: int = 1) -> None:
    """Sauvegarde une vidéo de visualisation."""

    nb_frames = len(list(mask_dir.glob(f"man_track*.tif")))

    # Obtenir les dimensions de la première image
    first_image = cv2.imread(str(img_dir / f"t{0:03d}.tif"), cv2.IMREAD_UNCHANGED)
    height, width = first_image.shape

    # Créer le VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))

    # Parcourir toutes les frames
    for frame_idx in range(nb_frames):
        image_path = img_dir / f"t{frame_idx:03d}.tif"
        mask_path = mask_dir / f"man_track{frame_idx:03d}.tif"  # ou "man_track{frame_idx:03d}.tif" selon votre cas
        # Générer la frame visualisée
        frame = make_gt_frame_color(image_path, mask_path,
                                         frame_idx=frame_idx,
                                         fontscale=fontscale,
                                         gamma_img=gamma_img,
                                         thickness=thickness)

        # Écrire la frame dans la vidéo
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Libérer le VideoWriter
    out.release()
    print(f"Vidéo sauvegardée à {output_path}")

if __name__ == '__main__':
    id_video = 13
    data_dir = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/HC_lab/moma_N_0_rechecked_cleaned/val/CTC")
    gt_video_path = Path("/home/hcourtei/Projects/Cell_proj/CellSam2Gilles/HC_lab/resul") / f"gt_video_{id_video}.mp4"
    img_dir = data_dir / f"{id_video:02d}"
    mask_dir = data_dir / f"{id_video:02d}_GT/TRA"

    scale = 4
    gamma_img = 2
    thickness = 1
    fontscale = 0.3
    video_fps = 10

    make_video_gt(img_dir, mask_dir, gt_video_path,
                  video_fps=video_fps,  # Frames par seconde
                  gamma_img=gamma_img,  # Paramètre gamma pour le contraste
                  fontscale=fontscale,  # Taille de la police pour les IDs
                  thickness=thickness)  # Épaisseur des contours

