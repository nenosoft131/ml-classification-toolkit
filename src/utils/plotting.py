import numpy as np
import plotly.colors
import pandas as pd
import seaborn as sns
from bokeh.plotting import figure, show, curdoc
from bokeh.models import HoverTool, TapTool, Range1d, BoxSelectTool, WheelZoomTool
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from matplotlib.colors import to_rgba

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# import src.constants as cfg

scatter_width = 1000
scatter_height = 600
lineplot_height = 500


def color_map(idx):
    import matplotlib.pyplot as plt
    return plt.get_cmap('tab20')(idx)


def rgba_to_str(color, alpha, value=0.8):
    color = to_rgba(color)
    cl = [c * value for c in color[:3]]
    return f"rgba({cl[0]}, {cl[1]}, {cl[2]}, {alpha})"


def create_plotly_colormap(class_names, class_colormap):
    label_name_to_id = {k: v for (k, v) in zip(class_names, range(len(class_names)))}
    mpl_colors = [class_colormap(label_name_to_id[k]) for k in class_names]
    colormap = {
        cls: rgba_to_str(color, 0.8, value = 0.9) for cls, color in zip(class_names,  mpl_colors)
    }
    return colormap


def create_bokeh_colormap(class_names, class_colormap):
    def to_bokeh(color):
        return [int(v*255) for v in color[:3]] + [color[3]]
    label_name_to_id = {k: v for (k, v) in zip(class_names, range(len(class_names)))}
    mpl_colors = [class_colormap(label_name_to_id[k]) for k in class_names]
    colormap = {
        cls: to_bokeh(color) for cls, color in zip(class_names,  mpl_colors)
    }
    return colormap


def make_scatter(source, label_field):

    assert(label_field in ['labelname', 'pred_labelname'])

    is_prediction = label_field == 'pred_labelname'

    TOOLS = "box_zoom, pan, reset"
    p = figure(
        width=scatter_width,
        height=scatter_height,
        tools=TOOLS,
    )

    color_field = 'colors' if is_prediction else 'pred_colors'

    opts = {'line_color': 'correct_color', 'line_width': 2} if is_prediction else {}
    if not is_prediction:
        opts.update({'legend_group': str(label_field)})

    p.circle(x='proj_x', y='proj_y', source=source,
             radius=0.6, selection_line_width=2,
             color=factor_cmap(label_field, list(cfg.color_map.values()), list(cfg.color_map.keys())),
             alpha=0.6,
             line_alpha=0.9,
             hover_alpha = 1.0,
             **opts,
             )

    tooltips = [
        ("", "$swatch:"+color_field),
        ("Class","@"+label_field),
        ("Filename", "@filename"),
        ("ID in file", "@id_in_file"),
    ]
    hover = HoverTool(tooltips=tooltips)
    p.tools.append(hover)

    box_select = BoxSelectTool()
    p.tools += [TapTool(), box_select]
    p.toolbar.active_drag = box_select

    p.legend.location = "top_left"

    # p.outline_line_width = 3
    # p.outline_line_alpha = 0.3
    # p.outline_line_color = "gray"
    p.xaxis.axis_line_width = 0
    p.yaxis.axis_line_width = 0
    # p.xgrid.grid_line_color = None
    # p.ygrid.grid_line_color = None
    p.xgrid.grid_line_alpha = 0.6
    p.xgrid.grid_line_dash = [5, 5]
    p.ygrid.grid_line_alpha = 0.6
    p.ygrid.grid_line_dash = [5, 5]

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

    p.toolbar.logo = None
    return p


def make_lineplot(freqs, height=lineplot_height, set_y_range=False):

    params = dict(
                  height=height,
                  background_fill_color="#fafafa",
                  sizing_mode='stretch_width',
                  x_range=Range1d(freqs[0], freqs[-1]),
                  # y_axis_type="log",
                  )

    if set_y_range:
        params['y_range'] = Range1d(0, cfg.RAMAN_YAXIS_MAX)

    p = figure(**params)

    hover = HoverTool(line_policy='next')
    hover.tooltips = [
        ("", "$swatch:colors"),
        ("Class","@labelname"),
        ("[id] File", "[@id_in_file] @filename"),
        # ("ID in file", "@id_in_file"),
        # ("y", "$y"),
    ]
    p.tools += [hover, TapTool()]
    p.xaxis.axis_label = 'Wave number [cm~1]'
    p.yaxis.axis_label = 'Intensity'
    p.xaxis.axis_label_text_font_size = "11pt"
    p.yaxis.axis_label_text_font_size = "11pt"
    p.yaxis.axis_label_standoff = 2
    p.xaxis.axis_label_standoff = 2
    p.toolbar.logo = None
    return p


def plot_raman(data, freqs=None, meta=None, class_column='target', color='viridis', options=None):
    if freqs is None:
        freqs = range(cfg.NUM_FREQUENCIES)

    if options is None:
        options = {}

    if isinstance(data, np.ndarray):
        X = data
    else:
        X = data[:len(freqs)].values

    fig = make_lineplot(freqs)
    fig = multi_line(fig, X, freqs, meta=meta, class_column=class_column, color=color, options=options)
    return fig


def multi_line(fig, data: np.ndarray, freqs, meta=None, class_column='target', color='viridis', options=None):
    if options is None:
        options = {}

    N, C = data.shape
    xs = [freqs for _ in range(N)]
    ys = [data[i] for i in range(N)]

    if meta is not None and color in meta:
        print(cfg.color_map)
        labels = list(meta[class_column].unique())
        label_name_to_id = {k: v for (k, v) in zip(labels, range(len(labels)))}
        colors = [cfg.color_map(label_name_to_id[cl]) for cl in meta[class_column]]
    else:
        colors = viridis(len(xs))

    s = ColumnDataSource({
        'xs': xs,
        'ys': ys,
        # 'colors': viridis(len(xs)),
        'colors': colors
    })

    if meta is not None:
        s.data.update(meta)
        if class_column in meta:
            s.data['labelname'] = meta[class_column]

    fig.multi_line(xs='xs', ys='ys', color='colors', source=s, **options)
    return fig


def make_confmat_display(y, y_pred, label_names, normalize=False):
    options = [
        ("Predicted classes (#)", None, 1, ''),
        ("Predicted classes (%)", 'true', 100, '.2f'),
    ]
    title, norm, scale, format = options[normalize]

    fig, ax = plt.subplots(1)
    cm = confusion_matrix(y, y_pred, normalize=norm)
    ax = sns.heatmap(cm*scale, annot=True, fmt=format, cmap=plt.cm.Blues)
    ax.set_yticklabels(label_names, rotation='horizontal')
    ax.set_xticklabels(label_names, rotation=45)#, size='small')
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig, cm


class TsnePlot():
    def __init__(self, colors):
        self.fig = go.Figure()
        self.colors = colors
        self.label_column = None

    def _scatter_plotly(self, meta, X_embedded, y_pred=None, y_correct=None):
        # data_dict = {
        #     'proj_x': X_embedded[:, 0],
        #     'proj_y': X_embedded[:, 1],
        #     'id': list(range(len(X_embedded[:, 0]))),
        #     'labelname': meta[self.label_column],
        #     # 'curve_index': y,
        #     # 'filename': meta['filename'],
        #     # 'case': meta['case'],
        #     # 'measurement': meta['measurement'],
        #     # 'source': meta['source']
        # }
        # df = pd.DataFrame(data_dict)
        df = pd.DataFrame(meta)
        df['proj_x'] = X_embedded[:, 0]
        df['proj_y'] = X_embedded[:, 1]
        df['id'] = list(range(len(X_embedded[:, 0])))
        df['labelname'] = meta[self.label_column]

        if y_pred is not None:
            # correct = y_pred == df['labelname']

            # if we are plotting a figure with predicted labels, show correct predictions with a green marker line
            correct_color = pd.Series(y_correct).map({True: 'green', False: 'red'})
            line_colors = [rgba_to_str(cl, 1.0, value=1.0) for cl in correct_color]
            colors = [rgba_to_str(self.colors[y], 0.8, value=0.9) for y in df['labelname']]
            line = dict(color=line_colors, width=1)
            marker = dict(opacity=0.9, color=colors, size=7, line=line, symbol='x')
            hover_template = ('Target: %{customdata[6]} => Pred.: %{customdata[3]}  %{customdata[4]} <br>'
                              'Species: %{customdata[0]}<br>'
                              'Culture: %{customdata[1]}<br>'
                              'Filename: %{customdata[2]}<br>'
                              'Source: %{customdata[5]}<br>'
                              # 'ID in file: %{customdata[2]}'
                              )
            z = np.zeros((len(df), 7), dtype=np.object_)
            z[:, 0] = df['labelname']
            z[:, 1] = df['case']
            z[:, 2] = df['filename']
            # z[:, 2] = df['id_in_file']
            z[:, 3] = y_pred
            z[:, 4] = pd.Series(y_correct).map({
                False: "!!!WRONG!!!",
                True: "",
            })
            z[:, 5] = df['source']
            z[:, 6] = df['target']
        else:
            # ground truth
            line_colors = [rgba_to_str(self.colors[l], 1.0) for l in df['labelname']]
            colors = [rgba_to_str(self.colors[l], 0.7) for l in df['labelname']]
            line = dict(color=line_colors, width=2)
            marker = dict(opacity=0.9, color=colors, size=2, line=line)
            hover_template = ('Species: %{customdata[0]}<br>'
                              'Culture: %{customdata[1]}<br>'
                              # 'Measurement: %{customdata[2]}<br>'
                              'Filename: %{customdata[3]}<br>'
                              'Source: %{customdata[4]}<br>'
                              # 'ID in file: %{customdata[2]}'
                              )
            z = np.zeros((len(df), 5), dtype=np.object_)
            z[:, 0] = df['labelname']
            z[:, 1] = df['case']
            z[:, 2] = df['measurement']
            z[:, 3] = df['filename']
            z[:, 4] = df['source']
            # z[:,2] = df['id_in_file']

        self.fig.add_trace(
            go.Scatter(
                mode='markers',
                x=df['proj_x'],
                y=df['proj_y'],
                marker=marker,
                showlegend=True,
                customdata=z,
                hovertemplate=hover_template
            )
        )


    def _scatter_plotly2(self, meta, X_embedded, y_pred=None, y_correct=None):
        classes = meta['class'].unique()
        for cls in classes:
            df = pd.DataFrame(meta)
            ids = df['class'] == cls
            df = df[ids]
            df['proj_x'] = X_embedded[ids, 0]
            df['proj_y'] = X_embedded[ids, 1]
            # df['id'] = list(range(len(X_embedded[:, 0])))
            # df['labelname'] = meta[self.label_column]

            # ground truth
            line_colors = rgba_to_str(self.colors[cls], 1.0)
            colors = rgba_to_str(self.colors[cls], 0.7)
            line = dict(color=line_colors, width=2)
            marker = dict(opacity=0.9, color=colors, size=2, line=line)
            hover_template = ('Species: %{customdata[0]}<br>'
                              'Culture: %{customdata[1]}<br>'
                              # 'Measurement: %{customdata[2]}<br>'
                              'Filename: %{customdata[3]}<br>'
                              'Source: %{customdata[4]}<br>'
                              # 'ID in file: %{customdata[2]}'
                              )

            z = np.zeros((len(df), 5), dtype=np.object_)
            z[:, 0] = df['class']
            z[:, 1] = df['case']
            z[:, 2] = df['measurement']
            z[:, 3] = df['filename']
            z[:, 4] = df['source']
            # z[:,2] = df['id_in_file']

            self.fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=df['proj_x'],
                    y=df['proj_y'],
                    marker=marker,
                    showlegend=True,
                    customdata=z,
                    hovertemplate=hover_template,
                    name=cls
                )
            )



    def plot(self, meta, X_embedded, label_column, pred=None, correct=None):

        self.label_column = label_column

        if pred is None:
            self._scatter_plotly2(meta, X_embedded)
        else:

            # train_idx = meta['split'] == 'train'
            # self._scatter_plotly(meta[train_idx], X_embedded[train_idx])

            # train_idx = meta['split'] == 'test'
            self._scatter_plotly2(meta, X_embedded, y_pred=pred, y_correct=correct)

        # FIXME: disable controls (line below has no effect)
        #  https://plotly.com/python/configuration-options/#removing-modebar-buttons
        # p.update_layout(modebar_remove=['zoom', 'pan'])


        # add 'fake' (invisible) plots to create legend entries
        # for label, color in self.colors.items():
        #     color_str = rgba_to_str(color, 1.0)
        #     self.fig.add_trace(go.Scatter(mode='lines', x=[0, 0], y=[0, 0], marker=dict(color=color_str), name=label))

        # fig.update_layout(plot_bgcolor='white')
        self.fig.update_layout(plot_bgcolor='#fcfcfc')
        self.fig.update_layout(autosize=True,
                               # width=scatter_width,
                               # height=scatter_height,
                               margin=dict(l=0, r=0, b=0, t=0, pad=0),
                               )
        self.fig.update_xaxes(visible=False)
        self.fig.update_yaxes(visible=False)
        # self.fig.update_yaxes(automargin=True)
        # self.fig.update_xaxes(automargin=True)


    def show(self):
        self.fig.show()