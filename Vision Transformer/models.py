import tensorflow as tf
from tensorflow.keras import layers


class Patches(layers.Layer):
    """
    This class will break images in patches (tokens) of size (batch_size, num_patch**2, channels).
    """
    def __init__(self, num_patches, d_model, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model

        self.projection = layers.Dense(d_model, activation="linear")

    def call(self, inputs):
        images_shape = tf.shape(inputs)
        batch_size, channels = images_shape[0], images_shape[1]
        patches = tf.image.extract_patches(inputs,
                                           sizes=[1, self.num_patches, self.num_patches, 1],
                                           strides=[1, self.num_patches, self.num_patches, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        reshaped_images = tf.reshape(patches, (batch_size, -1, channels))
        projection = self.projection(reshaped_images)
        return projection

    def config(self):
        config = super(Patches, self).config()
        config.update(num_patches=self.num_patches, d_model=self.d_model)
        return config


class ClassToken(tf.keras.layers.Layer):
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

    def call(self, inputs):
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


class Augmentation(layers.Layer):

    def __init__(self, flip="horizontal", rotation_factor=0.02, zoom_factor=0.2, **kwargs):
        super(Augmentation, self).__init__(**kwargs)
        self.normalization = layers.Rescaling(scale=1./255)
        self.flip = layers.RandomFlip(flip)
        self.rotation = layers.RandomRotation(factor=rotation_factor)
        self.zoom = layers.RandomZoom(height_factor=zoom_factor, width_factor=zoom_factor)

    def call(self, inputs, **kwargs):
        z = self.normalization(inputs, **kwargs)
        z = self.flip(z, **kwargs)
        z = self.rotation(z, **kwargs)
        return self.zoom(z, **kwargs)


def build_vit(input_shape, classes, n_encoders, num_patches, d_model,
              n_heads, mlp_dim=None, activation="gelu", dropout=0.2,
              to_augment=True, classification_head=(256, 128),
              return_attention_score=False):

    x = layers.Input(shape=input_shape)
    z = Augmentation()(x) if to_augment else x

    # patch the images into num_patches x num_patches and flatten them
    # patches = Patches(num_patches, d_model)(x)  # this can be substituted with Conv2D layer
    patches = layers.Conv2D(filters=d_model, kernel_size=num_patches, strides=num_patches, padding="VALID")(z)
    patches = layers.Reshape((patches.shape[1] * patches.shape[2], d_model))(patches)

    # Add class token and add positional embeddings.
    z = PatchEncoder()(patches)

    att_weights = []
    for i in range(n_encoders):
        z, weights = TransformerBlock(n_heads, d_model, mlp_dim, activation=activation, dropout=dropout)(z)
        att_weights.append(weights)

    y_attention = tf.stack(att_weights, axis=1, name="attention_stack")
    y = layers.LayerNormalization(epsilon=1e-6)(z)

    # extract the class token
    y = layers.Lambda(lambda v: v[:, 0])(y)
    for neurons in classification_head:
        y = layers.Dense(neurons, activation=activation)(y)
        y = layers.Dropout(dropout)(y)
    # compute the logits
    y = layers.Dense(classes)(y)

    outputs = y if not return_attention_score else [y, y_attention]
    return tf.keras.Model(inputs=x, outputs=outputs)
