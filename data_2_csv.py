import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_data_df(patients_path: Path) -> None:
    """
    Function to preparing data paths into csv files.

    Args:
        patients_path (Path): path to main directory of nifti files.
    """
    ct_paths = []
    pet_paths = []
    suv_paths = []
    masks_paths = []

    for root, _, files in os.walk(patients_path):
        for file in files:
            if 'CTres' in file:
                ct_paths.append(os.path.join(root, file))
            elif 'PET' in file:
                pet_paths.append(os.path.join(root, file))
            elif 'SUV' in file:
                suv_paths.append(os.path.join(root, file))
            elif 'SEG' in file:
                masks_paths.append(os.path.join(root, file))

    dataset_df = pd.DataFrame({'CT': ct_paths, 'PET': pet_paths, 'SUV': suv_paths, 'MASKS': masks_paths })
    training_dataset, test_dataset = train_test_split(dataset_df, test_size=0.25)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5)

    training_dataset.to_csv('data_csv/train_dataset.csv', index=False)
    val_dataset.to_csv('data_csv/val_dataset.csv', index=False)
    test_dataset.to_csv('data_csv/test_dataset.csv', index=False)

prepare_data_df('patients_path')