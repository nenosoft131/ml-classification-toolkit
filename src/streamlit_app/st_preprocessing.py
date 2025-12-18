import streamlit_app as st
import numpy as np
import pandas as pd

import config as cfg
import plotting
import util
import st_util
import preprocessing

NUM_SAMPLES = 6

st.set_page_config(layout="wide")

st.bokeh_chart = util.use_file_for_bokeh

dataset = st.selectbox(label='Select dataset', options=cfg.datasets, key='dataset')

# load spectroscopy data from disk
data, meta = st_util.cached_load_data(dataset)
data = data.iloc[:NUM_SAMPLES]
meta = meta.iloc[:NUM_SAMPLES]
freqs = data.columns.map(float)
X = data.values.astype(float)

class_names = list(meta['target'].unique())
data = data.values.astype(float)


"### Raw data"
"---"
st.bokeh_chart(plotting.plot_raman(data, freqs=freqs, meta=meta))


"### Despiking"
"---"

despiking_window = st.slider("Despike window", min_value=1, max_value=50, value=5)
despiking_threshold = st.slider("Despike threshold", min_value=1, max_value=50, value=10)
data = preprocessing.despike_whitaker(data, ma=despiking_window*2+1, threshold=despiking_threshold)

st.bokeh_chart(plotting.plot_raman(data, freqs=freqs, meta=meta))

"### Smoothing"
"Apply Savitzky-Golay filter."
"---"
smoothing_window = st.slider("Smoothing window", min_value=1, max_value=50, value=9)
smoothing_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2)
data = preprocessing.smooth_savitzky_golay(data, window=smoothing_window, polyorder=smoothing_polyorder)

st.bokeh_chart(plotting.plot_raman(data, freqs=freqs, meta=meta))


"### Baseline correction"
"Asymmetrically reweighted penalized least squares smoothing (arPLS)."
st.caption("Baek, S.J., et al. *Baseline correction using asymmetrically reweighted penalized least squares smoothing.* Analyst, 2015, 140, 250-257.")
"---"
exp_lam = st.slider("Smoothness 10^n", min_value=3, max_value=8, value=5)
baseline_lam = 10**exp_lam
st.write("= ", baseline_lam)
baseline_order = st.slider("Differential order", min_value=1, max_value=4, value=2)
data_baselined, baselines = preprocessing.baseline_correct(data, lam=baseline_lam, diff_order=baseline_order, return_baselines=True)

fig = plotting.plot_raman(data, freqs=freqs, meta=meta)
fig = plotting.multi_line(fig, baselines, freqs)
st.bokeh_chart(fig)

st.bokeh_chart(plotting.plot_raman(data_baselined, freqs=freqs, meta=meta))


"### Normalization"
"Normalization to unit vector length (L2)."
"---"
data = data_baselined
# data = preprocessing.demean(data)
data = preprocessing.normalize(data)
# data = preprocessing.baseline_correct(data)

st.bokeh_chart(plotting.plot_raman(data, freqs=freqs, meta=meta))
