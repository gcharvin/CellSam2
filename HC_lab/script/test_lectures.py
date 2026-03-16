
import tifffile
import numpy as np
import cv2
import torch
from PIL import Image


"""
 TRAINING lecture masque

 vos_raw_dataset.py
 
video_mask_root = self.train_dir / (video_name + "_GT") / "TRA"
segment_loader = CTCSegmentLoader(video_mask_root)

->  mask = tifffile.imread(mask_path)
"""



def analyse_mask(mask_path, data_version):
    mask = tifffile.imread(mask_path)

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]

    segments = {}
    for inst_id in instance_ids:
        segments[int(inst_id)] = torch.from_numpy(mask == inst_id)

    # Dilate background mask to avoid points touching objects to ensure there is no confusion between FPs being interpreted as objects
    kernel = np.ones((3, 3), np.uint8)
    bkgd_mask_dilated = cv2.erode((mask == 0).astype(np.uint8), kernel, iterations=2)  # Erode background = dilate objects
    segments['bkgd_mask'] = torch.from_numpy(bkgd_mask_dilated.astype(bool))

    print(f"MASK version", data_version)
    print(f"dtype {mask.dtype}, min {mask.min()}, max {mask.max()} unique {np.unique(mask)}")


print("-> MASK LOADING  ", "-"*20)
for data_version in ["moma_N_0_checked_uint8",  "moma_N_0_checked", "moma_N_0_rechecked"]:
    mask_path = f'/home/hcourtei/Projects/Cell_proj/data/{data_version}/moma/train/CTC/11_GT/TRA/man_track074.tif'
    analyse_mask(mask_path, data_version)
    # with tifffile.TiffFile(mask_path) as tif:
    #     page = tif.pages[0]
    #     print("dtype:", page.dtype)
    #     print("shape:", page.shape)
    #     print("BitsPerSample:", page.tags["BitsPerSample"].value)
    #     sample_format = page.tags.get("SampleFormat", None)
    #     if sample_format is None:
    #         print("SampleFormat: default (unsigned integer)")
    #     else:
    #         print("SampleFormat:", sample_format.value)


data_version = 'moma'
mask_path = f'/home/hcourtei/Projects/Cell_proj/data/moma/CTC/train/11_GT/TRA/man_track074.tif'
analyse_mask(mask_path, data_version)

"""
LECTURE IMAGE

modes PIL

Mode	Signification
L	    grayscale 8 bits
I;16	grayscale 16 bits
RGB	    RGB 8 bits



"""
def read_image(img_path, return_np=False):
    image = Image.open(img_path)
    image_mode = image.mode
    if image.mode == "RGB":
        pass
    elif image.mode == "I;16":
        # Convert to NumPy array
        arr = np.array(image)

        # Get 1st and 99th percentiles for robust scaling
        p1, p99 = np.percentile(arr, [1, 99])

        # Handle case where all values are identical (including all zeros)
        if p1 == p99:
            arr_8bit = np.zeros_like(arr, dtype=np.uint8)
        else:
            # Clip values to percentiles and scale to 0-255
            arr_clipped = np.clip(arr, p1, p99)
            arr_8bit = ((arr_clipped - p1) * (255.0 / (p99 - p1))).astype(np.uint8)

        # Convert to RGB by stacking
        image = Image.fromarray(arr_8bit).convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    else:
        raise ValueError(f"Unexpected image mode: {image.mode}. Please inspect and handle this mode manually.")

    if return_np:
        return np.array(image),  image_mode

    return image, image_mode



print("-> IMAGE LOADING ", "-"*20)

for data_version in ["moma_N_0_checked", "moma_N_0_checked_uint8", "moma_N_0_rechecked"]:
    img_path = f'/home/hcourtei/Projects/Cell_proj/data/{data_version}/moma/train/CTC/11/t074.tif'
    img, img_mode = read_image(img_path, return_np=True)
    print(" IMG version ", data_version)
    print(f'img_mode_pil {img_mode} dtype {img.dtype},shape {img.shape},min {img.min()},max {img.max()}') # image.mode == "RGB":
# /home/hcourtei/Projects/Cell_proj/data/moma_N_0_rechecked/train/CTC/11_GT/TRA


data_version = 'moma'
img_path = "/home/hcourtei/Projects/Cell_proj/data/moma/CTC/train/11/t074.tif"
img, img_mode= read_image(img_path, return_np=True)
print(" IMG version ", data_version)
print(f'img_mode_pil {img_mode} dtype {img.dtype},shape {img.shape},min {img.min()},max {img.max()}')# image.mode == "RGB":
