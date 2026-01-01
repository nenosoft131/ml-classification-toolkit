import json
import os
import pandas as pd
import itertools
from glob import glob
import numpy as np

# from src import config as cfg


fname_dataset = [
    "dataset1.txt",
    "datset2.txt"
]

data_def = {
    'f_dataset': {'fnames': fname_dataset, 'target': 'Fname'}
}


def load_data_old(data_root, groups):
    data = []
    filenames = []
    ids = []
    for clname in groups:
        group_data = []
        for fname in data_def[clname]['fnames']:
            df = pd.read_csv(os.path.join(data_root, fname), delimiter="\t", index_col=0)
            # df['filename'] = fname
            # df['id_in_file'] = df.index
            group_size = len(df)
            filenames.extend([fname]*group_size)
            ids.extend(list(df.index))
            group_data.append(df)
        group_data = pd.concat(group_data)

        # filenames = [[fname]*len(group_data) for fname in data_def[clname]['fnames']]
        # group_data['filename'] = list(itertools.chain(*filenames))
        data.append(group_data)
        data_def[clname]['counts'] = len(group_data)

    labels = [[g]*data_def[g]['counts'] for g in groups]
    labels = list(itertools.chain(*labels))

    targets = [[data_def[g]['target']]*data_def[g]['counts'] for g in groups]
    targets = list(itertools.chain(*targets))

    df_data = pd.concat(data).reset_index(drop=True)
    df_meta = pd.DataFrame({'group': labels, 'target': targets})
    df_meta['is_target'] = df_meta['target'].map(lambda t: 'no target' if t == 'notarget' else 'target')
    df_meta['filename'] = filenames
    df_meta['id_in_file'] = ids

    # return pd.concat([df_data, df_meta], axis=1)
    return df_data, df_meta


def read_file(filepath, format):
    if format == 'oem':
        d = np.loadtxt(filepath)
        return d[:, 2].T, d[:, 1]
    elif format == 'oem_oldform':
        d = np.loadtxt(filepath)
        return d[:, 3].T, d[:, 2]
    else:
        d = np.loadtxt(filepath)
        return d[1, :], d[0, :]


def read_single_spectrum(filepath, format, meta=None):
    fname = os.path.split(filepath)[1]
    spectrum_id = os.path.splitext(fname)[0]
    idx = int(fname.split('_')[-2][-2:])
    measurement = os.path.splitext(fname)[0][:-7]
    # reading files via numpy is much faster than via pandas
    data, frequencies = read_file(filepath, format)

    meta_ = dict(
        growth_time=None,
        substrate=None,
        strain=None,
    )
    meta_.update(
        dict(
            measurement=measurement,
            filename=fname,
            spectrum_id=spectrum_id,
        )
    )
    if meta is not None:
        meta_.update(meta)

    print(fname, measurement)
    return data, meta_, frequencies


def get_species_from_filename(fname: str, choices):
    for spec in choices:
        if spec in fname.lower():
            return spec
    return None


def read_folder(folder, sample_id, species_to_load):
    all_meta = []
    all_data = []
    all_frequencies = []
    # get metadata

    # try to read metadata JSON
    try:
        with open(os.path.join(folder, "metadata.json")) as json_file:
            meta = json.load(json_file)
    except:
        species = get_species_from_filename(folder, species_to_load)
        if species is None:
            return None, None, None
        meta = {
            "source": "NanoStruct",
            "species": species,
            "pcr": None
        }

    meta['sample'] = sample_id

    def reformat_pcr(item):
        if isinstance(item, str):
            item = [l.strip() for l in item.split(',')]
            item = "+".join(item)
        return item

    meta['pcr'] = reformat_pcr(meta['pcr'])

    for filepath in sorted(glob(os.path.join(folder, '*'))):
        # if os.path.isdir(filepath):
        #     read_folder(filepath, sample_id, species_to_load)
        if os.path.isfile(filepath) and os.path.splitext(filepath)[1] == '.txt':
            data_, meta_, frequencies_ = read_single_spectrum(filepath, meta=meta, format=format)
            all_meta.append(meta_)
            all_data.append(data_)
            all_frequencies.append(frequencies_)

    return all_data, all_meta, all_frequencies


def read_data(data_root, dataset, format='single', species_to_load=None):

    all_samples = []
    all_data = []
    all_frequencies = []
    data_path = os.path.join(data_root, dataset, '*')
    for folder in sorted(glob(data_path)):
        folder_name = os.path.split(folder)[1].split('_')
        if len(folder_name) == 2:
            datestr, sample_id = folder_name
        else:
            sample_id = folder_name[0]
        data, meta, frequencies = read_folder(folder, sample_id, species_to_load)
        if data is not None:
            all_samples.extend(meta)
            all_data.extend(data)
            all_frequencies.extend(frequencies)
        if format == 'oem':
            for filepath in glob(os.path.join(folder, 'OldForm', '*.txt')):
                data, meta, frequencies = read_single_spectrum(filepath, sample_id, format='oem_oldform')
                all_samples.append(meta)
                all_data.append(data)

    assert len(all_data) > 0, f"Could not read any data from path {data_root}."
    assert len(all_data) == len(all_samples), 

    df_meta = pd.DataFrame(all_samples)

    data = np.array(all_data)

    # print("Preprocessing data...")
    # import preprocessing
    # data = preprocessing.despike_whitaker(data)
    # data = preprocessing.smooth_savitzky_golay(data)
    # data = preprocessing.baseline_correct(data)
    # data = preprocessing.normalize(data)

    df_data = pd.DataFrame(data, columns=all_frequencies[0])
    df_data['spectrum_id'] = df_meta['spectrum_id']

    # df_data = pd.concat(all_dataframes[:10], ignore_index=True)
    # print(df_data.head())

    return df_data, df_meta


def load_dataset(dataset):
    path = os.path.join(cfg.data_root, dataset)
    assert os.path.isdir(path), f"Dataset path {path} not found!"
    if dataset == 'old_data':
        return load_data_old(data_root=path, groups=cfg.groups)
    elif dataset == '230913_OEM-Geraet':
        return read_data(data_root=path, format='oem')
    elif dataset == '240512_CommonRaman':
        df_meta = pd.read_csv(os.path.join(path, 'data_processed', 'metadata.csv'))
        df_data = pd.read_csv(os.path.join(path, 'data_processed', 'spectra.csv'))
        # return load_data(data_root=cfg.data_root, dataset=dataset)
        df_data = df_data.drop(columns='measurement')
        return df_data, df_meta
    else:
        return read_data(data_root=path, format='single')

