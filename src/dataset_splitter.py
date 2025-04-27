"""Module for load and split data"""

import typing as tp
from pathlib import Path

import numpy as np
from tqdm import tqdm
import os


def get_split_data(
    data_path: tp.Union[Path, str],
    valid_part: float = 0.2,
    test_part: float = 0.1,
) -> tp.Tuple[
    tp.Dict[str, tp.List[str]],
    tp.Dict[str, tp.List[str]],
    tp.Dict[str, tp.List[str]],
]:
    image_data_dict = {
        "Id": [],
        "Full_paths_img": [],
        "Full_paths_mask": [],
    }
    data_path = Path(data_path)
    for mask_file in tqdm(list(os.walk(data_path / 'masks'))[0][-1]):
        if mask_file.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
            continue
        id_img = mask_file.split('_')[0]
        image_data_dict['Id'].append(id_img)
        image_data_dict['Full_paths_mask'].append(str(data_path / 'masks' / mask_file))
        image_data_dict['Full_paths_img'].append(str(data_path / 'images' / f"{id_img}.jpg"))

    group_ids = image_data_dict['Id'].copy()
    np.random.shuffle(group_ids)
    train_indexes, val_indexes, test_indexes, _ = np.split(
        group_ids, [
            int((1 - valid_part - test_part) * len(group_ids)),
            int((1 - test_part) * len(group_ids)),
            len(group_ids)
        ]
    )
    print('Train, val, test: ', len(train_indexes), len(val_indexes), len(test_indexes))

    train_data_dict = {}
    validate_data_dict = {}
    test_data_dict = {}
    for fkey in image_data_dict:
        train_data_dict[fkey] = []
        validate_data_dict[fkey] = []
        test_data_dict[fkey] = []

    _data_dict = {
        0: train_data_dict,
        1: validate_data_dict,
        2: test_data_dict
    }
    for indexes_i, indexes in enumerate(
            (train_indexes, val_indexes, test_indexes)
    ):
        for index_i in indexes:
            for fkey in image_data_dict:
                _data_dict[indexes_i][fkey].append(
                    image_data_dict[fkey][int(index_i)]
                )
    print(
        'Train, val, test: ',
        len(train_data_dict['Id']),
        f"{train_data_dict['Id'][-1]} - {train_data_dict['Full_paths_img'][-1]} - {train_data_dict['Full_paths_mask'][-1]}",
        len(validate_data_dict['Id']),
        f"{validate_data_dict['Id'][-1]} - {validate_data_dict['Full_paths_img'][-1]} - {validate_data_dict['Full_paths_mask'][-1]}",
        len(test_data_dict['Id']),
        f"{test_data_dict['Id'][-1]} - {test_data_dict['Full_paths_img'][-1]} - {test_data_dict['Full_paths_mask'][-1]}",
    )
    return (
        train_data_dict,
        validate_data_dict,
        test_data_dict,
    )
