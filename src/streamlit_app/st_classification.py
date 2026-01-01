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
st.set_page_config(page_title="ML | Classification", page_icon=logo, layout='wide')

level_choices = ['bacterium', 'group', 'is_target']
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
freqs = data.columns[:-1].map(float)

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

X_raw = data.values[:, :-1].astype(float)
X = X_raw.copy()

with col[1]:
    with st.form("preprocessing_params"):
        st.markdown("### Pre-processing")

        if st.checkbox("Despiking"):
            X = preprocessing.despike_whitaker(X, threshold=50)

        if st.checkbox("Smoothing"):
            X = preprocessing.smooth_savitzky_golay(X)

        if st.checkbox("Baseline correction"):
            X = st_util.run_baseline_correction(X)

        if st.checkbox("Normalization (vector)"):
            X = preprocessing.normalize(X)

        st.form_submit_button('Update')


# Compute T-SNE projections
X_embedded = st_util.compute_tsne(X)

# Show T-SNE scatter plot with linked line plot. Selecting samples in scatter plot displays these samples
# in line plot.

# st.markdown("## Ground truth")
# "---"
# expander = st.expander("Ground truth", expanded=True)
# with expander:
p_scatter_gt, p_lines = util.create_interactive_tsne_plot(meta, X, label_column=level, X_embedded=X_embedded,
                                                          key='gt', freqs=freqs)

# st.markdown("## Predictions")
# "---"

with st.form("classification_params"):
    col_classifier = st.columns(4)
    with col_classifier[0]:
        st.markdown("### Classifier")
        method = st.selectbox(label='Select method', options=cfg.classification_methods)

    with col_classifier[0]:
        st.markdown("### Evaluation")
        num_cv_splits = st.slider(label='Num. cross-validation splits (default=5)', min_value=2, max_value=20, value=5)

    params = {
        'C': 1000,
        'num_trees': 10,
        'max_depth': 'inf',
        'alpha': 0.1,
        'boosting_depth': 1,
    }

    with col_classifier[1]:
        if method == cfg.METHOD_SVM:
            exp_params_svm = st.expander("Parameters SVM", expanded=True)
            with exp_params_svm:
                params['C'] = st.select_slider('C', options=[0.1, 1.0, 10, 100, 1000, 10000], value=params['C'])
        elif method == 'Random Forest':
            exp_params_rf = st.expander("Parameters Random Forest", expanded=True)
            with exp_params_rf:
                params['num_trees'] = st.select_slider('Num trees', options=[1, 5, 10, 100, 500], value=params['num_trees'])
                params['max_depth'] = st.select_slider('Num trees', options=[1, 10, 100, 'inf'], value='inf')
        if method == cfg.METHOD_PERCEPTRON:
            with st.expander("Parameters Perceptron", expanded=True):
                params['alpha'] = st.select_slider('alpha', options=[0.01, 0.1, 1.0, 10], value=params['alpha'])
        if method == cfg.METHOD_BOOSTING:
            with st.expander("Parameters Boosting", expanded=True):
                params['boosting_depth'] = st.select_slider('Depth', options=[1, 2, 3, 4, 5], value=params['boosting_depth'])
        else:
            pass

    if params['max_depth'] == 'inf':
        params['max_depth'] = None

    st.form_submit_button('Compute')

if len(selected_labels) < 2:
    st.error("The number of classes has to be greater than one.")
else:
    print(f"Running classification with method {method}")

    y_pred, cl_result = run_classification(X, y, method=method, params=params, num_cv_splits=num_cv_splits)

    with st.expander("Predictions", expanded=True):
        st.caption(f"{method=}, {params=}")
        p_scatter_pred, p_lines_pred = util.create_interactive_tsne_plot(meta, X=X, y_pred=y_pred, label_column=level,
                                                                         X_embedded=X_embedded, key='pred', freqs=freqs)

        p_confusion_matrix, confmat = plotting.make_confmat_display(y, y_pred, selected_labels)
        p_confusion_matrix_norm, confmat_norm = plotting.make_confmat_display(y, y_pred, selected_labels, normalize=True)

        # Create table with classwise classification report
        multi_confmat = multilabel_confusion_matrix(y, y_pred)
        # print(multi_confmat)
        rows = {}
        for class_id, class_name in enumerate(selected_labels):
            tn, fp, fn, tp = multi_confmat[class_id].ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            rows[class_name] = [sensitivity, specificity]
        df_results = pd.DataFrame.from_dict(rows, orient='index', columns=['Sensitivity', 'Specificity'])
        df_results.index.names = ['Class']
        df_results = pd.concat([df_results, df_results.mean().to_frame('Mean').T])

        " "
        " "
        columns = st.columns(4, gap='large')
        with columns[0]:
            p_confusion_matrix
        with columns[1]:
            p_confusion_matrix_norm
        with columns[2]:
            st.markdown("### Cross-validation performance")
            # result_box = st.columns([2, 2, 1])
            # with result_box[0]:
            st.markdown("#### Results per class")
            df_results *= 100
            st.dataframe(df_results.style.format("{:.1f}%"))
            # st.dataframe(df_results, column_config={'Sensitivity': st.column_config.NumberColumn("foo", format="%f")})
            # with result_box[1]:
        with columns[3]:
            st.markdown("###   ")
            st.markdown("###   ")
            st.markdown("#### Total (unbalanced)")
            # st.markdown("Accuracy")
            st.metric("Accuracy", value=f"{accuracy_score(y, y_pred) * 100:.1f}%")
            # style_metric_cards()

    # export data and results
    os.makedirs('results', exist_ok=True)
    np.savetxt('./results/spectra_raw.txt', X_raw)
    np.savetxt('./results/spectra_proc.txt', X)
    np.savetxt('./results/tsne.txt', X_embedded)
    np.savetxt('./results/confmat.txt', confmat, fmt='%d')
    np.savetxt('./results/confmat_norm.txt', confmat_norm, fmt='%.2f')

    # label_names = list(meta[label_column].unique())
    # label_name_to_id = {k: v for (k, v) in zip(label_names, range(len(label_names)))}
    # y = meta[label_column].map(label_name_to_id)

    data_dict = {
        'id': list(range(len(X_embedded[:, 0]))),
        'tsne_x': X_embedded[:, 0],
        'tsne_y': X_embedded[:, 1],
        'filename': meta['filename'],
        # 'id_in_file': meta['id_in_file'],
        'label': meta[label_column],
    }
    if y_pred is not None:
        pred_labelnames = [label_id_to_name[l] for l in y_pred]
        corrects = y_pred == y
        data_dict.update({
            'pred_labelname': pred_labelnames,
            'correct': corrects,
        })

    df_predictions = pd.DataFrame(data_dict)
    df_predictions.to_csv ('./results/predictions.csv', index=False)
