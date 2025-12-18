import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import streamlit.components.v1 as components
# from streamlit_plotly_events import plotly_events

from bokeh.io import output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import show, figure, save

# from . import plotting
# from .. import config as cfg


# patch streamlit_app bokeh_chart function to be compatible with bokeh >2.4.1
def use_file_for_bokeh(chart: figure, chart_height=500):
    output_file('bokeh_graph.html')
    save(chart)
    with open("bokeh_graph.html", 'r', encoding='utf-8') as f:
        html = f.read()
    components.html(html, height=chart_height)


js_callback_code="""
    const inds = cb_obj.indices
    const ind = inds[0]

    const x = [];
    const y = [];
    for (let i = ind*1024; i < (ind+1)*1024; i++)
    {
        y.push(X[i]);
    }

    for (let i =0; i < 1024; i++)
    {
        x.push(i);
    }
    console.log('Tap event occurred at x-position: ' + inds[0])
    source.data = { x: x, y: y }
"""


def update_line_plot_plotly(p, ids, X: np.ndarray, meta: pd.DataFrame, freqs):
    if ids is None or len(ids) == 0:
        return

    p.renderers = []
    # freqs = df.columns[:cfg.NUM_FREQUENCIES].map(float)

    print(f"Ids: {ids}")
    # indices = []
    # lines = []
    # for curve_idx, point_idx in zip(curve_ids, ids):
    #     s = df.query("(curve_index == @curve_idx) & (point_index == @point_idx)")
    #     if len(s.index.values) > 0:
    #         indices.append(s.index.values[0])
    #         lines.append(s[:cfg.NUM_FREQUENCIES].values)

    selected_data = meta.loc[ids]
    # print(selected_data)
    # X = meta.iloc[:cfg.NUM_FREQUENCIES].values

    # If level was changed, the current selection becomes invalid. In this case, skip drawing lines.
    # FIXME: this should be handled by a callback on the level selectbox that invalidates the old selection
    # if len(lines) == len(ids):
    #     selected_data = df.loc[indices]

    selected_data['xs'] = [freqs for _ in ids]
    selected_data['ys'] = [X[i] for i in ids]

    opts = dict(line_alpha=0.4, hover_line_alpha=1.0, line_width=1, hover_line_width=1)
    p.multi_line(xs='xs', ys='ys', color='colors', source=ColumnDataSource(selected_data), **opts)


def update_line_plot_bokeh(p, ids, df: pd.DataFrame, curve_ids=None):
    X = df.iloc[:cfg.NUM_FREQUENCIES].values
    freqs = df.columns[:cfg.NUM_FREQUENCIES].map(float)

    print(f"Ids: {ids}")
    selected_data = df.loc[ids]
    selected_data['xs'] = [freqs for _ in ids]
    selected_data['ys'] = [X[i] for i in ids]

    p.renderers = []
    opts = dict(line_alpha=0.4, hover_line_alpha=1.0, line_width=1, hover_line_width=1)
    p.multi_line(xs='xs', ys='ys', color='colors', source=ColumnDataSource(selected_data), **opts)



def create_interactive_tsne_plot(meta, X, label_column, y_pred=None, X_embedded=None, key=None, orientation='horizontal', freqs=None):
    assert(orientation in ['horizontal', 'vertical'])

    if freqs is None:
        freqs = range(X.shape[1])

    label_names = list(meta[label_column].unique())
    label_name_to_id = {k: v for (k, v) in zip(label_names, range(len(label_names)))}
    label_id_to_name = {v: k for (k, v) in zip(label_names, range(len(label_names)))}

    y = meta[label_column].map(label_name_to_id)

    data_dict = {
        'proj_x': X_embedded[:, 0],
        'proj_y': X_embedded[:, 1],
        'id': list(range(len(X_embedded[:, 0]))),
        'labelname': meta[label_column],
        'colors': [cfg.color_map(id) for id in y],
        'curve_index': y,
        'filename': meta['filename'],
        'culture': meta['culture'],
        'measurement': meta['measurement'],
        # 'id_in_file': meta['id_in_file'],
    }
    if y_pred is not None:
        pred_labelnames = [label_id_to_name[l] for l in y_pred]
        corrects = y_pred == y
        correct_colors = ['green' if correct else 'red' for correct in corrects]
        data_dict.update({
            'pred_labelname': pred_labelnames,
            'pred_colors': [cfg.color_map(id) for id in y_pred],
            'correct': corrects,
            'correct_color': correct_colors,
        })

    df_data_scatter = pd.DataFrame(data_dict)
    p_lines = plotting.make_lineplot(freqs)

    # df_data_lines = pd.concat([data, df_data_scatter], axis=1)
    # point_indices = np.array([np.arange(0, num_samples) for num_samples in df_data_scatter['labelname'].value_counts().values], dtype="object")
    # point_indices = np.concatenate(point_indices).ravel()
    # df_data_lines['point_index'] = point_indices
    # print(df_data_scatter)

    if cfg.backend_scatter == 'plotly':
        p_scatter = plotting.make_scatter_plotly(df_data_scatter, 'labelname')
        if orientation == 'horizontal':
            columns = st.columns(2)
            with columns[0]:
                selected_points = plotly_events(p_scatter, select_event=True, key=key)
        else:
            selected_points = plotly_events(p_scatter, select_event=True, key=key)

        ids = [p['pointIndex'] for p in selected_points]
        valid_ids = [id for id in ids if id in df_data_scatter.index]

        # st.write(selected_points)
        # st.write(ids)
        # st.write(valid_ids)

        update_line_plot_plotly(p_lines, ids=valid_ids, X=X, meta=df_data_scatter, freqs=freqs)

        if orientation == 'horizontal':
            with columns[1]:
                st.bokeh_chart(p_lines)
        else:
            st.bokeh_chart(p_lines)
    else:
        s1 = ColumnDataSource(data=data_dict)
        p_scatter = plotting.make_scatter(s1, 'labelname')

        javascript = False
        if not javascript:
            def callback(attr, old, new):
                update_line_plot_bokeh(p_lines, ids=new, df=df_data_lines)
            s1.selected.on_change('indices', callback)
        else:
            from bokeh.models import CustomJS
            xs = [freqs for _ in [0]]
            ys = [X[i] for i in [0]]
            s2 = ColumnDataSource({'x': xs, 'y': ys,})
            p_lines.multi_line(xs='xs', ys='ys', source=s2)
            s1.selected.js_on_change('indices', CustomJS(args=dict(source=s2, X=X), code=js_callback_code))

    return p_scatter, p_lines


def init_random(seed=0):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def bool_str(x):
    return str(x).lower() in ['True', 'true', '1']

