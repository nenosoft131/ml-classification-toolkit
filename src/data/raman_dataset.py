import numpy as np
import pandas as pd
from typing import Callable
import torch.utils.data as td

from src.data import preprocessing


class RamanDataset(td.Dataset):
    _repr_indent = 4

    def __init__(self,
                 meta_csv: str,
                 data_csv: str,
                 target_species: list[str],
                 transform: Callable | None = None,
                 balance_by: str = None,
                 max_samples_per_class: int = None,
                 filter: dict = None,
                 query: str = None,
                 sample_n: int = None,
                 **kwargs):

        self.transform = transform

        self.meta_csv = meta_csv
        self.data_csv = data_csv
        self.balance_by = balance_by
        self.max_samples_per_class = max_samples_per_class
        self.filter = filter
        self.query = query
        self.sample_n = sample_n
        self.targets = None
        self.target_species = target_species

        self._load_data()
        self.data = self._preprocess(self.data)

    def __len__(self):
        return len(self.data)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = preprocessing.despike_whitaker(X)
        X = preprocessing.smooth_savitzky_golay(X)
        X = preprocessing.baseline_correct(X)
        X = preprocessing.normalize(X)
        return X

    def _load_data(self) -> None:
        # load metadata from csv
        assert self.meta_csv is not None
        self.metadata = pd.read_csv(self.meta_csv)

        # self.metadata = self._filter_data(self.metadata, dict(species=self.target_species))

        # if there is no ground-truth 'species' annotation, use 'pcr' data
        def fill_species_label(df: pd.DataFrame) -> pd.DataFrame:
            df['species_or_pcr'] = df['species']
            species_missing = pd.isna(df['species_or_pcr'])
            df.loc[species_missing, 'species_or_pcr'] = df['pcr'][species_missing]
            return df

        self.metadata = fill_species_label(self.metadata)
        # self.metadata['species_id'] = self.metadata['species_or_pcr'].apply(self.species_to_categorical)

        # A sample can contain multiple (bacteria) species, so convert single class
        # target (1 in k) to multi-label target (n in k). Create one new column for each target_species.
        def create_multiclass_labels(df, label_column, target_classes):
            targets = np.zeros((len(df), len(target_classes)))
            for idx, cl in enumerate(target_classes):
                targets[:, idx] =  self.metadata[label_column].apply(lambda value: cl in value).values
            return targets.astype(np.float32)

        self.species_labels = create_multiclass_labels(
            self.metadata,
            'species_or_pcr',
            self.target_species
        )

        # remove items without target
        has_target = np.max(self.species_labels, axis=1).astype(bool)
        self.species_labels = self.species_labels[has_target]
        self.metadata = self.metadata[has_target]

        # load Raman spectra from CSV and store data as numpy array
        assert self.data_csv is not None
        data_df = pd.read_csv(self.data_csv)
        self.data = data_df.merge(self.metadata['spectrum_id'], on='spectrum_id').drop(
            columns='spectrum_id').values.astype(np.float32)


    def _filter_data(self, data: pd.DataFrame, filter: dict = None, query: str = None) -> pd.DataFrame:
        if filter is not None:
            for feature_name, keep_values in filter.items():
                if isinstance(keep_values, str):
                    keep_values = [keep_values]
                data = data.query(f"{feature_name} in {list(keep_values)}")
        if query is not None:
            if not isinstance(query, str):
                raise ValueError(f"Parameter 'query' must be string: {query}")
            data = data.query(query)
        return data

    def species_to_categorical(self, species_name: str) -> int:
        if not getattr(self, 'species_to_id_map', None):
            class_ids = range(len(self.target_species))
            self.species_to_id_map = {name: idx for name, idx in zip(self.target_species, class_ids)}

        return self.species_to_id_map[species_name]

    def __getitem__(self, idx) -> dict:

        sample = {
            'spectrum' : self.data[idx].reshape(1, -1),
            'species_labels': self.species_labels[idx].reshape(1, -1),
            'target_species_names': self.target_species
        }
        # sample.update(self.metadata.iloc[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    from torchvision.transforms import v2
    import torch
    from src.data import transforms as t

    transform = v2.Compose([
        t.DespikeWhitaker(),
        t.SmoothSavitzkiGoilay(),
        t.BaselineCorrect(),
        t.Normalize(),
        v2.ToDtype(torch.float32)
    ])

    ds = RamanDataset(
        data_csv="../../data/processed/spectra.csv",
        meta_csv="../../data/processed/splits/train-split1.csv",
        transform=transform,
        target_species=['brevis', 'plantarum']
    )

    print(ds[0])
    exit()
    dl = td.DataLoader(ds, batch_size=5, shuffle=False, num_workers=0)

    for batch in dl:
        print(batch)
