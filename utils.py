from pathlib import Path
import os
from typing import Union
import torch

mcm_data_path = Path('data', 'McM')
kodak_data_path = Path('data', 'Kodak')

mcm_patches_path = Path('data', 'McM_patches')
kodak_patches_path = Path('data', 'Kodak_patches')

mcm_mosaic_patches_path = Path('data', 'McM_mosaic_patches')
kodak_mosaic_patches_path = Path('data', 'Kodak_mosaic_patches')

pickle_data_path = Path('data', 'pickle_data')
best_early_stopping_model_path = Path('best_early_stopping_model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_files_from_dir_with_pathlib(dir_name: Union[str, Path]) -> 'list[Path]':
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[Path]
    """
    return sorted(list(Path(dir_name).iterdir()))


def empty_dir(dir_path: Path):
    """Erase all the files inside the directory.

    :param dir_path: Path of the directory.
    :type dir_path: Path
    """
    filepaths = get_files_from_dir_with_pathlib(dir_path)
    for f in filepaths:
        os.remove(f)