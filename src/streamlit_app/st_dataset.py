import os

import streamlit_app as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from PIL import Image

import preprocessing
import st_util
import classification
import config as cfg
import plotting
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import util
# from streamlit_extras.metric_cards import style_metric_cards

cfg.backend_scatter = 'plotly'
st.bokeh_chart = util.use_file_for_bokeh

logo = Image.open(cfg.FILEPATH_LOGO_JPG)
st.set_page_config(page_title="NanoStruct | Classification", page_icon=logo, layout='wide')

level_choices = ['target', 'group', 'is_target']
if 'level' not in st.session_state:
    print("init level========================")
    st.session_state['level'] = level_choices[0]
    level_index = 0
else:
    level_index = level_choices.index(st.session_state['level'])


@st.cache
def run_classification(X, y, method, num_cv_splits, params):
    y_pred, cl_result = classification.classify_data(X, y, method=method, params=params, num_cv_splits=num_cv_splits)
    return y_pred, cl_result

#
# Setup main panel
#
st.markdown("## Classification")
st.markdown(f"Train model and evaluate using cross-validation.")
"---"

# with st.form("preprocessing_params"):
col = st.columns(4)
with col[0]:
    st.markdown("### Data")
    dataset = st.selectbox(label='Select dataset', options=cfg.datasets, key='dataset')

# load spectroscopy data from disk
data, meta = st_util.cached_load_data(dataset)
freqs = data.columns.map(float)

# select level in target hierarchy and specific classes within this level
with col[0]:
    level = st.selectbox(label='Select label type', options=level_choices, index=level_index, key='level')
    label_column = level
    all_label_names = list(meta[label_column].unique())
    # create colors
    cfg.colors = {name: cfg.color_map(i) for i, name in enumerate(all_label_names)}
    selected_labels = st.multiselect('Select training classes', all_label_names, all_label_names)

label_name_to_id = {k: v for (k, v) in zip(selected_labels, range(len(selected_labels)))}
label_id_to_name = {v: k for (k, v) in zip(selected_labels, range(len(selected_labels)))}

# filter by selected classes
keep_ids = meta[label_column].isin(selected_labels)
data = data[keep_ids]
meta = meta[keep_ids].reset_index()
y = meta[label_column].map(label_name_to_id)

X_raw = data.values.astype(float)
X = X_raw.copy()

# Compute T-SNE projections
X_embedded = st_util.compute_tsne(X)


# Create T-SNE plot
p_scatter_gt, p_lines = util.create_interactive_tsne_plot(meta, X, label_column=level, X_embedded=X_embedded,
                                                          key='gt', freqs=freqs)

