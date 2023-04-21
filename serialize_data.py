import numpy as np
import pickle
import shutil
from pathlib import Path
from typing import MutableMapping, Union, Optional
import imageio as iio

from utils import empty_dir, get_files_from_dir_with_pathlib, \
    mcm_data_path, kodak_data_path, mcm_patches_path, kodak_patches_path, \
        mcm_mosaic_patches_path, kodak_mosaic_patches_path, pickle_data_path


def serialize_features_and_classes(features_and_classes: MutableMapping[Union[str, Path],
                                                                      Union[np.ndarray, int]],
                                   pickle_path: str) \
        -> None:
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    :param pickle_path: Path of the pickle file
    :type pickle_path: str
    """
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(features_and_classes, pickle_file)
    

def handle_one_file(input_img_path: Union[str, Path],
                    input_mosaic_img_path: Union[str, Path],  
                    output_file_path: Union[str, Path]) -> None:
    features_and_classes = {}

    # gather feature and info
    img = np.transpose(np.array(iio.imread(input_img_path)),(2,0,1))
    mosaic_img = np.transpose(np.array(iio.imread(input_mosaic_img_path)),(2,0,1))

    # assign value for dict
    features_and_classes['img'] = img
    features_and_classes['mosaic_img'] = mosaic_img
    
    # save object as a pickle
    serialize_features_and_classes(features_and_classes, output_file_path)


def create_pickle_data(dataset: str) -> None:
    """Serialize all the data in a data split
    
    :param split: The data split
    :type split: str
    """
    if dataset == "McM":
        img_patches_path = mcm_patches_path
        img_mosaic_patches_path = mcm_mosaic_patches_path
    elif dataset == "Kodak":
        img_patches_path = kodak_patches_path
        img_mosaic_patches_path = kodak_mosaic_patches_path
    
    pickle_dir = Path.joinpath(pickle_data_path, dataset)
    empty_dir(pickle_dir)

    img_patches = get_files_from_dir_with_pathlib(img_patches_path)
    for img_file in img_patches:
        img_name = img_file.name
        img_mosaic_file = Path.joinpath(img_mosaic_patches_path, img_name)
        pickle_file = Path.joinpath(pickle_dir, img_file.stem+".pickle")
        handle_one_file(img_file, img_mosaic_file, pickle_file)


def main():
    create_pickle_data("McM")
    create_pickle_data("Kodak")


if __name__ == "__main__":
    main()