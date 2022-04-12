import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
