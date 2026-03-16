from PIL import Image
import numpy as np
import cv2



def read_image(img_path, gamma_img=0.7):
    """Lit une image 16-bit et la normalise pour un bon affichage en niveaux de gris"""
    image = Image.open(img_path)
    if image.mode == "I;16":
        arr = np.array(image)
        # Meilleure distribution des niveaux de gris
        p1, p99 = np.percentile(arr, [1, 99])
        if p1 == p99:
            return np.zeros_like(arr, dtype=np.uint8)
        # Normalisation avec étirement de contraste
        arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        # Appliquer un gamma pour améliorer le contraste
        arr_norm = np.power(arr_norm/255.0, gamma_img) * 255.0
        return arr_norm.astype(np.uint8)
    elif image.mode == "L":
        return np.array(image)
    else:
        raise ValueError(f"Mode d'image non supporté: {image.mode}")



def get_centroid(mask: np.ndarray, cell_id: int) -> tuple:
    """Calcule le centroïde d'une cellule dans un masque."""
    try:
        cell_mask = (mask == cell_id).astype(np.uint8)
        moments = cv2.moments(cell_mask)
        if moments["m00"] != 0:
            cx, cy =  int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        else:
            cx,cy = -1,-1
    except Exception as e:
        print(f"Erreur dans get_centroid: {e}")
    return cx,cy