from mosaic_bayer import mosaic_bayer
from pathlib import Path
import imageio as iio
import numpy as np

from utils import empty_dir, get_files_from_dir_with_pathlib, mcm_patches_path, kodak_patches_path, mcm_mosaic_patches_path, kodak_mosaic_patches_path


def mosaic_patches(patch_path, mosaic_pattern='rggb'):

    # Get data name
    data_name = patch_path.name
    if data_name == 'McM_patches':
        mosaic_patch_path = mcm_mosaic_patches_path
    else:
        mosaic_patch_path = kodak_mosaic_patches_path

    # Get image paths
    patch_paths = get_files_from_dir_with_pathlib(patch_path)

    # Create mosaiced patches
    for patch_path in patch_paths:
        img = np.array(iio.imread(patch_path), dtype=np.uint8)
        _, mosaic, _ = mosaic_bayer(img, mosaic_pattern)
        mosaic_patch_name = patch_path.stem + '.tif'
        mosaic_patch_file_path = Path(mosaic_patch_path, mosaic_patch_name)
        iio.imwrite(mosaic_patch_file_path, mosaic.astype(np.uint8))


def main():
    # Empty directories
    empty_dir(mcm_mosaic_patches_path)
    empty_dir(kodak_mosaic_patches_path)

    # Create mosaiced patches
    mosaic_patches(mcm_patches_path)
    mosaic_patches(kodak_patches_path)


if __name__ == '__main__':
    main()