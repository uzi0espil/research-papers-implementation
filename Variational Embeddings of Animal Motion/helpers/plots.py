import pandas as pd
from typing import Union, List
from typing_extensions import Literal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_signals(X: Union[pd.DataFrame, np.ndarray], columns=None, title=""):
    if isinstance(X, pd.DataFrame):
        index = X.index
        columns = X.columns if columns is None else columns
        X = X.values
    else:
        index = np.arange(X.shape[0])
        columns = [f"Column {i}" for i in range(X.shape[-1])] if columns is None else columns

    fig = go.Figure()
    for i, column in enumerate(columns):
        fig.add_trace(go.Scatter(x=index, y=X[:, i], name=column))
    fig.update_layout(title=title)
    return fig


def plot_embedding_space(X: np.ndarray,
                         clusters: Union[None, str, List] = None,
                         dims: int = 2,
                         reduction_algo: Literal['pca', 'tsne', 'umap'] = "umap",
                         **kwargs):
    def get_reduction_algo():
        if reduction_algo.lower() == "tsne":
            return TSNE
        elif reduction_algo.lower() == "pca":
            return PCA
        elif reduction_algo.lower() == "umap":
            return UMAP
        else:
            raise ValueError("Only TSNE, PCA and UMAP reduction algorithms are supported.")

    if dims not in [2, 3]:
        raise ValueError("Only reduction to 2 or 3 dimensions are possible.")

    algo = get_reduction_algo()
    X_trans = algo(n_components=dims, **kwargs).fit_transform(X)
    df_ = pd.DataFrame(X_trans, columns=[f"Component {i + 1}" for i in range(dims)])

    plt_kwargs = dict(z="Component 3") if dims == 3 else dict()
    scatter = px.scatter if dims == 2 else px.scatter_3d
    fig = scatter(df_, x="Component 1", y="Component 2", color=clusters, **plt_kwargs)
    return fig


def learning_curve(histories):
    histories = [histories] if not isinstance(histories, list) else histories

    histories_count = len(histories)
    subplot_titles = ['learning curve iteration: {}'.format(i) for i in range(histories_count)]

    fig = make_subplots(rows=histories_count, cols=1,
                        shared_xaxes=True,
                        subplot_titles=subplot_titles)

    for i, history in enumerate(histories):
        plot_val_loss = True
        if 'val_loss' not in history.history:
            plot_val_loss = False

        x_axis = np.arange(0, len(history.history['loss']))

        fig.add_trace(go.Scatter(x=x_axis, y=history.history['loss'],
                                 name='training {}'.format(i), line=dict(color='blue'), mode='lines'),
                      row=i + 1, col=1)
        if plot_val_loss:
            fig.add_trace(go.Scatter(x=x_axis, y=history.history['val_loss'],
                                     name='validation {}'.format(i), line=dict(color='red'), mode='lines'),
                          row=i + 1, col=1)

    fig.update_layout(height=300 * histories_count, title_text="Learning curves")

    return fig


def plot_signals_cluster(df, clusters=None, columns=None, latent_code=None, title=""):
    columns = df.columns if columns is None else columns
    if clusters is not None:
        clusters = pd.Series(clusters, dtype="category")
        df = df[-len(clusters):]

    n_rows = len(columns) if latent_code is None else len(columns) + 1
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True)

    for i, col in enumerate(columns):
        a_fig = px.scatter(df, x=df.index, y=col, color=clusters)
        for d in a_fig.data:
            fig.add_trace(go.Scatter(x=d['x'], y=d['y'], legendgroup=d.legendgroup, mode=d.mode,
                                     marker=dict(color=d.marker.color, symbol=d.marker.symbol),
                                     name="{}, cluster: {}".format(col, d.name), hovertemplate=d.hovertemplate),
                          row=i + 1, col=1)
        fig.update_yaxes(title_text=col, row=i + 1, col=1)
    if latent_code is not None:
        fig.add_trace(go.Heatmap(z=latent_code.T, x=df.index, showscale=False, connectgaps=False), row=n_rows, col=1)
        fig.update_yaxes(title_text="Latent code", row=n_rows, col=1)
    fig.update_layout(title=title)
    return fig
