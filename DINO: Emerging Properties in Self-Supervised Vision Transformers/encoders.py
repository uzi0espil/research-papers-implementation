import tensorflow as tf
from keras import layers
import numpy as np

from typing import Tuple, Optional, List


class ClassToken(layers.Layer):
    """Append a class token to an input layer."""

    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self._channels = None
        self._cls_embeddings = None

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self._channels = input_shape[-1]
        self._cls_embeddings = tf.Variable(
            name="ClassToken",
            initial_value=cls_init(shape=(1, 1, self._channels), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self._cls_embeddings, [batch_size, 1, self._channels]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        return config


class PatchEncoder(layers.Layer):
    """
    Patch Encoder task is as following:

    - Project the flattened patches to patch embeddings
    - Add positional learned embeddings to the patch embeddings.

    Positional embeddings are learned through 1D position embeddings.
    """

    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)

        self.positional_embeddings = None
        self.positions = None
        self.class_token = ClassToken()

    def build(self, inputs_shape):
        self.positional_embeddings = layers.Embedding(inputs_shape[1] + 1, output_dim=inputs_shape[2])
        self.positions = tf.range(0, limit=inputs_shape[1] + 1, delta=1)  # adding the class

    def call(self, inputs, **kwargs):
        patch_embeddings = self.class_token(inputs)  # add class token.
        positional_embeddings = self.positional_embeddings(self.positions, **kwargs)
        return patch_embeddings + positional_embeddings  # add positional embeddings


class MLPHead(layers.Layer):

    def __init__(self, mlp_dim=None, activation="gelu", dropout=0.2, **kwargs):
        super(MLPHead, self).__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.dropout_rate = dropout

        self.layer1 = None
        self.dropout = None
        self.layer2 = None

    def build(self, input_shape):
        mlp_dim = self.mlp_dim if self.mlp_dim is not None else input_shape[-1]
        self.layer1 = layers.Dense(mlp_dim, activation=self.activation)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer2 = layers.Dense(input_shape[-1])  # linear with same shape as features

    def call(self, inputs, **kwargs):
        out = self.layer1(inputs, **kwargs)
        out = self.dropout(out, **kwargs)
        out = self.layer2(out, **kwargs)
        return self.dropout(out, **kwargs)


class TransformerBlock(layers.Layer):

    def __init__(self, n_heads, d_model, mlp_dim=None, activation="gelu", dropout=0.2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()
        self.mha = layers.MultiHeadAttention(n_heads, d_model)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLPHead(mlp_dim=mlp_dim, activation=activation, dropout=dropout)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, **kwargs):
        z = self.norm1(inputs, **kwargs)
        z, weights = self.mha(z, z, return_attention_scores=True)
        z = self.dropout(z, **kwargs)
        z = self.add([z, inputs])
        out = self.norm2(z, **kwargs)
        out = self.mlp(out, **kwargs)
        out = self.add([out, z])
        return out, weights


def build_vit(input_shape: Tuple[int, int, int],
              n_encoders: int,
              n_patches: int,
              d_model: int,
              n_heads: int,
              mlp_dim: Optional[List[int]] = None,
              activation: str = "gelu",
              dropout: float = 0.1,
              return_attention_score: bool = False):
    """
    Build Vision Transformer model based on the paper.

    :param input_shape: Image dimension [W, H, C]
    :param n_encoders: Number of transformer encoder blocks
    :param n_patches: How many patches to break the image, this will then translate to patch_size.
    :param d_model: The constant D dimension across the transformer projections.
    :param n_heads: Number of heads in MHA.
    :param mlp_dim: List of neurons for the mlp head of each encoder. at the end an additional dense layer is added with
                    neurons equal to d_model.
    :param activation: encoder's MLP activations.
    :param dropout: dropout rate across all dropout layers in the transformer.
    :param return_attention_score: Whether to return the attention scores or not.
                                   It has shape of [n_encoders, n_head, patch_size + 1, patch_size + 1]
    :return: Keras Model.
    """
    if input_shape[0] % n_patches != 0:
        raise ValueError("Patches should evenly divide input image.")

    x = layers.Input(shape=input_shape)
    patch_size = int(x.shape[1] / np.sqrt(n_patches))

    # patch the images into num_patches x num_patches and flatten them
    # patches = Patches(num_patches, d_model)(x)  # this can be substituted with Conv2D layer
    patches = layers.Conv2D(filters=d_model, kernel_size=patch_size, strides=patch_size, padding="VALID")(x)
    patches = layers.Reshape((patches.shape[1] * patches.shape[2], d_model))(patches)

    # Add class token and add positional embeddings.
    z = PatchEncoder()(patches)

    att_weights = []
    for i in range(n_encoders):
        z, weights = TransformerBlock(n_heads, d_model, mlp_dim, activation=activation, dropout=dropout)(z)
        att_weights.append(weights)

    y_attention = tf.stack(att_weights, axis=1, name="attention_stack")
    y = layers.LayerNormalization(epsilon=1e-6)(z)

    # extract the [cts] embeddings only.
    y = layers.Lambda(lambda v: v[:, 0])(y)

    outputs = y if not return_attention_score else [y, y_attention]
    return tf.keras.Model(inputs=x, outputs=outputs, name="ViT")
