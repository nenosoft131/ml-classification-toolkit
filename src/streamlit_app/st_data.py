import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from data import data_def
from bokeh.models import Band, ColumnDataSource
import plotly.express as px

import plotting
import config as cfg
import util
from st_util import cached_load_data, compute_tsne

def filter(meta, data, column, values):
    keep_ids = meta[column].isin(values)
    data = data[keep_ids]
    meta = meta[keep_ids]
    return data, meta


st.bokeh_chart = util.use_file_for_bokeh

logo = Image.open(cfg.FILEPATH_LOGO_JPG)
st.set_page_config(page_title="ML | Data", page_icon=logo, layout='wide')


"# Data viewer"
"---"

dataset = st.selectbox(label='Select dataset', options=cfg.datasets, key='dataset')

# load spectroscopy data from disk
data, meta = cached_load_data(dataset)
freqs = data.columns.map(float)

all_species = list(meta['species'].unique())
selected_species = st.multiselect('Select species', all_species, all_species)
data, meta = filter(meta, data, 'species', selected_species)

all_cultures = list(meta['culture'].unique())
selected_culture = st.multiselect('Select culture', all_cultures, all_cultures)
data, meta = filter(meta, data, 'culture', selected_culture)

if dataset == 'old_data':
    with st.expander("Data config"):
        st.write(data_def)

with st.expander("Loaded dataframe"):
    st.dataframe(data)

st.markdown(f"### Total number of spectra: {len(data)}")


#
# Plot raw, unprocessed data
#
"\n"
st.markdown("## Raw data \n ---")

# col = st.columns([1,1,3])
# with col[0]:
    # level_choices = ['species', 'is_target']
    # level = st.selectbox(label='Select label type', options=level_choices, index=0, key='level')
level =  'species'
# st.markdown(f"Number of samples per class")

class_names = list(meta[level].unique())
label_name_to_id = {k: v for (k, v) in zip(class_names, range(len(class_names)))}

plotly_colormap = plotting.create_plotly_colormap(class_names, cfg.color_map)
bokeh_colormap = plotting.create_bokeh_colormap(class_names, cfg.color_map)

#
# Bar chart with class counts
#
columns = st.columns(3)
with columns[0]:
    "#### Number of spectra per species"
    counts = pd.DataFrame(meta['species'].value_counts(sort=False))
    fig = px.bar(x=counts.index,
                 y=counts['species'],
                 color=counts.index,
                 color_discrete_map=plotly_colormap,
                 height=300,
                 width=600,
                 text_auto=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis={'title': "Count"},
        xaxis={'title': "Class"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

with columns[1]:
    "#### Number of cultures per species"
    counts = pd.DataFrame(meta.groupby('species')['culture'].nunique())
    fig = px.bar(x=counts.index,
                 y=counts['culture'],
                 color=counts.index,
                 color_discrete_map=plotly_colormap,
                 height=300,
                 width=600,
                 text_auto=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis={'title': "Count"},
        xaxis={'title': "Class"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


opts = dict(line_alpha=0.4, hover_line_alpha=1.0, line_width=1, hover_line_width=1)
# p_raw = plotting.plot_raman(data, freqs=freqs, meta=meta, options=opts, class_column=level, color=level)
# st.bokeh_chart(p_raw)

# columns = st.columns(2)
# with columns[0]:
#
# Plot mean over entire dataset
#
st.markdown("### Mean over entire dataset")

p_mean = plotting.make_lineplot(freqs, set_y_range=False)
y_mean = data.mean()
y_std = data.std()
p_mean.line(freqs, y_mean, line_dash=(10, 7), line_width=2)
source = ColumnDataSource({
    'base': freqs,
    'lower': y_mean - y_std,
    'upper': y_mean + y_std
})

band = Band(base='base', lower='lower', upper='upper', source=source,
            fill_alpha=0.3, fill_color="yellow", line_color="black")
p_mean.add_layout(band)
st.bokeh_chart(p_mean)


#
# Plot species means
#

st.markdown("### Species means")
p_class_means = plotting.make_lineplot(freqs, set_y_range=False)

col = st.columns([1,1,3])
with col[0]:
    offset_delta = st.number_input("Line offsets", value=3)

y_offset = {cls: i*offset_delta for i, cls in enumerate(class_names)}

for clname in class_names:

    print(clname)
    class_data = data[meta[level] == clname]
    y_mean = class_data.mean() + y_offset[clname]
    y_std = class_data.std()
    y_lower = y_mean - y_std
    y_upper = y_mean + y_std

    source = ColumnDataSource({
        'base': freqs,
        'mean': y_mean,
        'lower': y_lower,
        'upper': y_upper
    })

    p_class_means.line(x='base', y='mean', line_width=2, source=source,
                       color=bokeh_colormap[clname], legend_label=clname)

st.bokeh_chart(p_class_means)

#
# Plot culture means
#

st.markdown("### Culture means")
p_sample_means = plotting.make_lineplot(freqs, set_y_range=False)

groups = meta.groupby(['culture'])
group_means = []

for group_name, ids in groups.indices.items():

    clname = meta.iloc[ids[0]].species
    class_data = data.iloc[ids]

    y_mean = class_data.mean()
    y_std = class_data.std()
    y_lower = y_mean - y_std
    y_upper = y_mean + y_std

    group_means.append(y_mean)

    source = ColumnDataSource({
        'base': freqs,
        'mean': y_mean + y_offset[clname],
        'lower': y_lower,
        'upper': y_upper
    })

    p_sample_means.line(x='base', y='mean', line_width=1, source=source,
                       color=bokeh_colormap[clname], legend_label=group_name)

st.bokeh_chart(p_sample_means)


" "
"## TSNE visualization"
"---"

# X = data.iloc[:, :cfg.NUM_FREQUENCIES].values.astype(float)
# X_embedded = compute_tsne(X)
# p_scatter_gt, p_lines = util.create_interactive_tsne_plot(
#     meta, X, label_column=level, X_embedded=X_embedded, key='gt', freqs=freqs)

X = np.array(group_means)
X_embedded = compute_tsne(X)
sample_meta = meta.groupby(['culture', 'species', 'measurement']).size().reset_index()
sample_meta['filename'] = ''
p_scatter_gt2, p_lines2 = util.create_interactive_tsne_plot(
    sample_meta, X, label_column='species', X_embedded=X_embedded, key='foo', freqs=freqs)

