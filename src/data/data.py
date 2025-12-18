import json
import os
import pandas as pd
import itertools
from glob import glob
import numpy as np

# from src import config as cfg


fname_ecoli = [
    "Target1_230615_EColiNis/230615_Data_Map3.txt"
]
fname_ecoli_ringer = [
    "Target1_230720_RingerLoesungEColiNis/PL02-XX_MutaFl_NoAl2O3_Str1_Points_40xObj_OD1_1x5s.txt",
    "Target1_230720_RingerLoesungEColiNis/PL02-XX_MutaFl_NoAl2O3_Str2_Points_40xObj_OD1_1x5s.txt",
    "Target1_230720_RingerLoesungEColiNis/PL02-XX_MutaFl_NoAl2O3_Str3_Points_40xObj_OD1_1x5s.txt"
]
fname_brevis_ringer = [
    "Target2_230720_RingerLoesungBrevis/PL02-XX_WuHo_NoAl2O3_Str1_Points_40xObj_OD1_1x5s.txt",
    "Target2_230720_RingerLoesungBrevis/PL02-XX_WuHo_NoAl2O3_Str2_Points_40xObj_OD1_1x5s.txt",
    "Target2_230720_RingerLoesungBrevis/PL02-XX_WuHo_NoAl2O3_Str3_Points_40xObj_OD1_1x5s.txt"
]
fname_notarget_dirty = [
    "NoTarget_230715_Dirty/OK12-24_Str4_40xObj_OD1_1x5s.txt",
    "NoTarget_230715_Dirty/OK12-24_Str4_40xObj_OD1_1x5s_2.txt",
    "NoTarget_230715_Dirty/OK12-24_Str6_40xObj_OD1_1x5s.txt",
    "NoTarget_230715_Dirty/OK12-24_Str6_40xObj_OD1_1x5s_2.txt",

    "NoTarget_230715_Dirty/OK12-25_Str4_40xObj_OD1_1x5s.txt",
    "NoTarget_230715_Dirty/OK12-25_Str4_40xObj_OD1_1x5s_2.txt",
    "NoTarget_230715_Dirty/OK12-25_Str6_40xObj_OD1_1x5s.txt",
    "NoTarget_230715_Dirty/OK12-25_Str6_40xObj_OD1_1x5s_2.txt",
]

fname_notarget_noanalyte = [
    "NoTarget_230720_NoAnalyte/PL02-XX_3nmAl2O3_Str1_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_NoAl2O3_Str1_Points_40xObj_OD1_1x5s_2.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_3nmAl2O3_Str2_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_NoAl2O3_Str1_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_3nmAl2O3_Str3_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_NoAl2O3_Str2_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_NoAl2O3_Str1_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_NoAnalyte/PL02-XX_NoAl2O3_Str3_Points_40xObj_OD1_1x5s.txt"
]

fname_notarget_ringer = [
    "NoTarget_230720_RingerLoesung/PL02-XX_Ringer_3nmAl2O3_Str1_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_RingerLoesung/PL02-XX_Ringer_3nmAl2O3_Str2_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_RingerLoesung/PL02-XX_Ringer_3nmAl2O3_Str3_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_RingerLoesung/PL02-XX_Ringer_NoAl2O3_Str1_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_RingerLoesung/PL02-XX_Ringer_NoAl2O3_Str2_Points_40xObj_OD1_1x5s.txt",
    "NoTarget_230720_RingerLoesung/PL02-XX_Ringer_NoAl2O3_Str3_Points_40xObj_OD1_1x5s.txt",
]

fname_ecoli_new = [
    "Target1_230803_EColiNis/PLXX_285Ag_EColi2xZent_Map3_40xObj_OD1_1x15s.txt",
    "Target1_230803_EColiNis/PLXX_285Ag_EColi2xZent_Map4_40xObj_OD1_1x15s.txt",
    "Target1_230803_EColiNis/PLXX_30Ag_EColi2xZent_Map5_40xObj_OD1_1x15s.txt",
    "Target1_230803_EColiNis/PLXX_45Ag15Au_EColi2xZent_Map1_40xObj_OD1_1x15s.txt",
    "Target1_230803_EColiNis/PLXX_45Ag15Au_EColi2xZent_Map2_40xObj_OD1_1x15s.txt",
]

fname_omnibio = [
    "Target3_230803_OmniBioLactos/PLXX_285Ag_OmniBio2xZent_Map8_40xObj_OD1_1x15s.txt",
    "Target3_230803_OmniBioLactos/PLXX_30Ag_OmniBio2xZent_Map6_40xObj_OD1_1x15s.txt",
    "Target3_230803_OmniBioLactos/PLXX_45Ag15Au_OmniBio2xZent_Map7_40xObj_OD1_1x15s.txt",
]

data_def = {
    'ecoli': {'fnames': fname_ecoli, 'target': 'ecoli'},
    'ecoli_ringer': {'fnames': fname_ecoli_ringer, 'target': 'ecoli'},
    'brevis_ringer': {'fnames': fname_brevis_ringer, 'target': 'brevis'},
    'ecoli_new': {'fnames': fname_ecoli_new, 'target': 'ecoli'},
    'omnibio': {'fnames': fname_omnibio, 'target': 'omnibio'},
    'notarget_dirty': {'fnames': fname_notarget_dirty, 'target': 'notarget'},
    'notarget_noanalyte': {'fnames': fname_notarget_noanalyte, 'target': 'notarget'},
    'notarget_ringer': {'fnames': fname_notarget_ringer, 'target': 'notarget'}
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

    # meta['species'] = make_list(meta['species'])
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
    assert len(all_data) == len(all_samples), "Number of meta data entries must match number of data samples"

    df_meta = pd.DataFrame(all_samples)
    # df_meta['is_target'] = df_meta['target'].map(lambda t: 'no target' if t == 'NoTarget' else 'target')

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

