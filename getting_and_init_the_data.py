from pathlib import Path
from typing import Optional, Union
from dataset_class import MosaicImageDataset
from torch.utils.data import DataLoader, random_split
import torch

from utils import pickle_data_path


def get_dataset(data_split: str,
                data_dir: Union[str, Path],
                load_into_memory: Optional[bool] = True) \
        -> MosaicImageDataset:
    return MosaicImageDataset(
                    data_split=data_split,
                    data_dir=data_dir,
                    load_into_memory=load_into_memory)


def get_data_loader(dataset: MosaicImageDataset,
                    batch_size: int,
                    shuffle: bool,
                    drop_last: bool) \
        -> DataLoader:
    """Creates and returns a data loader.

    :param dataset: Dataset to use.
    :type dataset: dataset_class.MyDataset
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :return: Data loader, using the specified dataset.
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                      drop_last=drop_last, num_workers=1)


def get_all_data_loaders(batch_size: int):
    mcm_dataset = get_dataset('McM', pickle_data_path)
    
    # Set seed for pytorch
    torch.manual_seed(0)

    # Split the data to train and test with the ratio of 80/20
    train_size = int(0.8 * len(mcm_dataset))
    test_size = len(mcm_dataset) - train_size
    train_dataset, test_dataset = random_split(mcm_dataset, [train_size, test_size])

    # Split train data to train and validation data with ratio 3/1
    train_size = int(0.75 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    

    # Create the data loaders
    train_loader = get_data_loader(train_dataset, batch_size, True, False)
    val_loader = get_data_loader(val_dataset, batch_size, False, False)
    test_loader = get_data_loader(test_dataset, batch_size, False, False)

    return train_loader, val_loader, test_loader


def main():
    pass


if __name__ == "__main__":
    main()