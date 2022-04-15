import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2


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


def attention_mask(image, attention):
    """
    Compute the attention map as stated in section D7 and based on the paper: https://arxiv.org/pdf/2005.00928.pdf

    :param image: The original image.
    :param attention: the attention score of the image, it has shape (n_encoders, n_heads, patch_size+1, patch_size+1)
    :return: attended image.
    """
    # take the mean over all heads
    attention = attention.mean(axis=1)

    # take care of the residual
    ## add to identity matrix
    attention = attention + np.eye(attention.shape[-1])
    ## sum it for every encoder
    attention_sum = attention.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]
    ## divide attention with attention sum per each encoder.
    attention = attention / attention_sum

    # Recursively multiply the weights
    attention_map = attention[-1]  # take last encoder's attention
    for encoder_att in attention[::-2]:  # reverse and skip last
        attention_map = np.matmul(attention_map, encoder_att)

    patch_size = int(np.sqrt(attention.shape[-1] - 1))  # remove token class and take the root
    # build the attention from the class token
    mask = attention_map[0, 1:].reshape(patch_size, patch_size)
    normalized_mask = mask / mask.max()
    # resize to the inputs
    mask = cv2.resize(normalized_mask, (image.shape[1], image.shape[0]))
    mask = mask[..., np.newaxis]  # add a channel

    image = (mask * image).astype("uint8")

    return image


def attention_image(image, attention, **kwargs):
    """
    Plot the original image and the attended image side by side

    :param image: the original image.
    :param attention: the attention score of the image, it has shape (n_encoders, n_heads, patch_size+1, patch_size+1)
    :param kwargs: Plotly layout kwargs.
    :return: Plotly figure
    """
    attended_image = attention_mask(image, attention)
    fig = make_subplots(1, 2, subplot_titles=("Original", "Attention Map"))
    fig.add_trace(go.Image(z=image), 1, 1)
    fig.add_trace(go.Image(z=attended_image), 1, 2)
    fig.update_layout(**kwargs)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
