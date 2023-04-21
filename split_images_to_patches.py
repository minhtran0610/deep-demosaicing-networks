from pathlib import Path
import imageio as iio
import numpy as np

from utils import get_files_from_dir_with_pathlib, empty_dir, mcm_data_path, kodak_data_path, mcm_patches_path, kodak_patches_path



def create_patch(img_path: Path, patch_size):
    # Get data name
    data_name = img_path.parent.name
    if data_name == 'McM':
        patches_path = mcm_patches_path
    else:
        patches_path = kodak_patches_path

    # Read image
    img = iio.imread(img_path)
    img = img.astype(np.float32)
    img_name = img_path.stem

    # Create patches
    m = 0
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            # Rotate and save patch
            if patch.shape == (patch_size, patch_size, 3):
                for k in range(4):
                    patch_rot = np.rot90(patch, k)
                    patch_name = f'{img_name}_{m}_{k}.tif'
                    iio.imwrite(Path(patches_path, patch_name), patch_rot.astype(np.uint8))
                m += 1


def main():
    # Empty directories
    empty_dir(mcm_patches_path)
    empty_dir(kodak_patches_path)

    # Get image paths
    mcm_img_paths = get_files_from_dir_with_pathlib(mcm_data_path)
    kodak_img_paths = get_files_from_dir_with_pathlib(kodak_data_path)

    # Create patches
    for img_path in mcm_img_paths:
        create_patch(img_path, 33)
    for img_path in kodak_img_paths:
        create_patch(img_path, 33)


if __name__ == '__main__':
    main()