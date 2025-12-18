import os
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit

from src.data import data
import autoroot

import logging

log = logging.getLogger(__name__)


def create_csv(
        data_csv: str | os.PathLike | bytes,
        meta_csv: str | os.PathLike | bytes,
        input_dir,
        dataset_name: str,
        species_to_load: list[str]
):
    df_data, df_meta = data.read_data(data_root=input_dir,
                                      dataset=dataset_name,
                                      species_to_load=species_to_load)

    os.makedirs(os.path.split(data_csv)[0], exist_ok=True)
    os.makedirs(os.path.split(meta_csv)[0], exist_ok=True)

    df_data.to_csv(data_csv, index=False)
    df_meta.to_csv(meta_csv, index=False)


def split_dataset(meta_csv: str | os.PathLike | bytes,
                  output_dir: str | os.PathLike | bytes,
                  n_splits: int,
                  train_size: float,
                  random_state: int = 0) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Load metadata file.
    metadata_df = pd.read_csv(meta_csv)

    # Split data into train, validation and test.
    # train, val = train_test_split(metadata_df, 'image_name', test_split=0.20, split_by='case_id', seed=0)
    # train, val, _, _ = train_test_split(range(len(metadata_df)), y, train_size=0.8, stratify=y)

    gss = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)

    for i, (train_index, test_index) in enumerate(gss.split(range(len(metadata_df)), metadata_df['sample'], metadata_df['sample'])):
        print(f"Fold {i}:")
        # print(f"  Train: index={train_index}, group={y[train_index]}")
        # print(f"  Test:  index={test_index}, group={y[test_index]}")

        # Save split data as CSV file.
        split_dict = {
            'train': train_index,
            'val': test_index,
        }
        for k, v in split_dict.items():
            filepath = os.path.join(output_dir, f"{k}-split{i+1}.csv")
            split_df = metadata_df.iloc[v]
            print(split_df.head())
            split_df.to_csv(filepath, index=False)


def preprocess_data():
    pass


@hydra.main(version_base=None, config_path="../../configs", config_name="prepare.yaml")
def make_dataset(cfg: DictConfig) -> None:

    create_csv(
        data_csv=os.path.join(cfg.output_dir, cfg.filename_data),
        meta_csv=os.path.join(cfg.output_dir, cfg.filename_meta),
        input_dir=cfg.input_dir,
        dataset_name=cfg.dataset,
        species_to_load=cfg.species
    )

    split_dataset(
        meta_csv=os.path.join(cfg.output_dir, cfg.filename_meta),
        output_dir=os.path.join(cfg.output_dir, cfg.split.out_dirname),
        n_splits=cfg.split.n_splits,
        train_size=cfg.split.train_size,
        random_state=cfg.seed
    )


if __name__ == "__main__":
    make_dataset()