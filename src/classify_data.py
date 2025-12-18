import os
from lib2to3.pytree import convert

import numpy as np
import pandas as pd
import classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
# from streamlit_app import st_util
from sklearn.manifold import TSNE
from ast import literal_eval

from src.data import preprocessing
from utils import plotting, util
import constants as NC

import streamlit as st
from bokeh.plotting import show
from bokeh.models import ColumnDataSource
from streamlit_plotly_events import plotly_events
st.bokeh_chart = util.use_file_for_bokeh


species_for_radeberger_experiment = ['casei', 'rossiae', 'plantarum', 'buchneri', 'lindneri']


def color_map(idx):
    import matplotlib.pyplot as plt
    return plt.get_cmap('tab20')(idx)


def preprocess(X):
    X = preprocessing.despike_whitaker(X)
    X = preprocessing.smooth_savitzky_golay(X)
    X = preprocessing.baseline_correct(X)
    X = preprocessing.normalize(X)
    return X


def plot_tsne(meta_train, X_train, meta_test=None, X_test=None, y_pred=None):
    print("Computing TSNE embeddings...")
    # X = data.drop(columns='spectrum_id').values
    X = X_train
    if X_test is not None:
        X = np.vstack([X_train, X_test])
    X_embedded = TSNE(
        n_components=2,
        learning_rate='auto',
        random_state=0,
        perplexity=min(30.0, len(X) - 1)
    ).fit_transform(X)

    X_embedded_train = X_embedded[:len(X_train)]

    # metadata_df = pd.concat([meta_train, meta_test])
    # metadata_df['split'] = ['train'] * len(X_train) + ['test'] * len(X_test)

    p_scatter = plotting.TsnePlot(colors)
    p_scatter.plot(meta_train, X_embedded_train, label_column='class')

    if X_test is not None:
        X_embedded_test = X_embedded[:len(X_test)]
        p_scatter.plot(meta_test, X_embedded_test, label_column='class', pred=y_pred, correct=correct)

    return p_scatter.fig


def split_by_case(metadata_df, train_sources=None):
    gss = GroupShuffleSplit(n_splits=3, train_size=0.8, random_state=0)
    splits = gss.split(range(len(metadata_df)), metadata_df['case'], metadata_df['case'])
    train_idx, test_idx = next(splits)
    train_set = metadata_df.iloc[train_idx]
    test_set = metadata_df.iloc[test_idx]
    # for i, (train_index, test_index) in enumerate(splits):
    #     print(f"Fold {i}:")
    #     # print(f"  Train: index={train_index}, group={y[train_index]}")
    #     # print(f"  Test:  index={test_index}, group={y[test_index]}")
    #
    #     # Save split data as CSV file.
    #     split_dict = {
    #         'train': train_index,
    #         'val': test_index,
    #     }
    #     for k, v in split_dict.items():
    #         split_df = metadata_df.iloc[v]
    #         print(split_df.head())
    return train_set, test_set


def split_by_source(metadata_df, train_sources, test_sources=None):
    if test_sources is None:
        # set test sources to complement of train sources
        all_sources = metadata_df['source'].unique()
        test_sources = list(set(all_sources) - set(train_sources))

    train_set = metadata_df[metadata_df['source'].isin(train_sources)]
    test_set = metadata_df[metadata_df['source'].isin(test_sources)]
    return train_set, test_set


def make_list(item):
    if isinstance(item, str):
        item = literal_eval(item)
        item = [s.strip() for s in item]
    return item


def select_first_species(item):
    return item.split('+')[0]


def create_detector_target(df, target_class):
    df['target'] = df['class']

    # unique_targets = list(df['target'].unique())
    # target_name_to_id = {k: v for (k, v) in zip(unique_targets, range(len(unique_targets)))}
    # target_id_to_name = {v: k for (k, v) in zip(unique_targets, range(len(unique_targets)))}

    # consider one class as 'positive' and combine all other classes into 'negative'
    def merge_classes(x):
        if x == target_class:
            return 1
        else:
            return -1

    df['target'] = df['target'].apply(merge_classes)
    return df


def train_model(meta_train, X_train):
    print("Fitting model...")
    y_train = meta_train['target']
    clf = classification.get_model(NC.METHOD_SVM, params={'C': 1000.0})
    clf.fit(X_train, y_train)
    print("Done.")
    return clf


data_root = '/home/browatbn/dev/nanostruct/ML/data'
dataset = '240512_CommonRaman'
split = 'split1'
data_dir = os.path.join(data_root, 'processed')
split_dir = os.path.join(data_root, 'processed', 'splits')

# Load spectra
data = pd.read_csv(os.path.join(data_dir, 'spectra.csv'))

# Load metadata file
meta_train = pd.read_csv(os.path.join(split_dir, 'train-split1.csv'))
meta_test  = pd.read_csv(os.path.join(split_dir, 'val-split1.csv'))

# data = data[:1000]
# metadata_df = metadata_df[:1000]

freqs = data.columns[:-1].map(float)


# if there is no ground-truth 'species' annotation, use 'pcr' data
def fill_species_label(df: pd.DataFrame) -> pd.DataFrame:
    df['species_or_pcr'] = df['species']
    species_missing = pd.isna(df['species_or_pcr'])
    df.loc[species_missing, 'species_or_pcr'] = df['pcr'][species_missing]
    return df

meta_train = fill_species_label(meta_train)
meta_test = fill_species_label(meta_test)

# unique_labels = meta_train['class'].unique()
unique_labels = NC.ALL_SPECIES
label_name_to_id = {k: v for (k, v) in zip(unique_labels, range(len(unique_labels)))}
label_id_to_name = {v: k for (k, v) in zip(unique_labels, range(len(unique_labels)))}

colors = {label: color_map(label_name_to_id[label]) for label in unique_labels}

# filter training data
# with_species = species_for_radeberger_experiment
# metadata_df = metadata_df.query("`class` in @with_species")

# remove unnecessary colors/labels
colors = {k: v for k,v in colors.items() if k in unique_labels}

# consider one class as 'positive' and combine all other classes into 'negative'
# metadata_df = create_detector_target(metadata_df, target_class='casei')

# create target for multiclass classification
def create_classification_target(df):

    def select_first_item(item):
        return item.split('+')[0]

    def to_categorical(label):
        return label_name_to_id[label]

    _target = df['species_or_pcr']
    _target = _target.apply(select_first_item)
    _target = _target.apply(to_categorical)
    df['target'] = _target
    return df

meta_train = create_classification_target(meta_train)
meta_test = create_classification_target(meta_test)

# Create splits
# meta_train, meta_test = split_by_source(metadata_df, train_sources=['NanoStruct'])

X_train = data.merge(meta_train['spectrum_id'], on='spectrum_id').drop(columns='spectrum_id').values
X_test = data.merge(meta_test['spectrum_id'], on='spectrum_id').drop(columns='spectrum_id').values

print("Preprocessing data...")
X_train = preprocess(X_train)
X_test = preprocess(X_test)


if True:
    clf = train_model(meta_train, X_train)

    print("\nTraining accuracy:")
    y_pred = clf.predict(X_train)
    print(accuracy_score(meta_train['target'], y_pred))

    print("\nTest accuracy:")
    y_pred = clf.predict(X_test)
    # print(y_pred)
    # pred_targets = [target_id_to_name[y] for y in y_pred]
    # tp = 0
    # correct = np.zeros_like(y_pred, dtype=bool)
    # for i, (pred, gts) in enumerate(zip(y_pred, meta_test['target'])):
    #     if pred == gts:
    #         correct[i] = True
    correct = meta_test['target'] == y_pred
    acc = float(np.sum(correct)) / len(y_pred)
    print(f"{acc=}")

    p_scatter = plot_tsne(meta_train, X_train, meta_test, X_test, y_pred)

p_scatter = plot_tsne(meta_train, X_train)
p_scatter.show()

# make clickable

if False:
    selected_points = plotly_events(p_scatter, select_event=True, key=None)
    ids = [p['pointIndex'] for p in selected_points]
    # valid_ids = [id for id in ids if id in df_data_scatter.index]

    X = data.drop(columns='spectrum_id').values
    p_lines = plotting.make_lineplot(freqs)

    util.update_line_plot_plotly(p_lines, ids, X, metadata_df, freqs)
    st.bokeh_chart(p_lines)

