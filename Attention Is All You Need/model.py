import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert(self.d_model % self.num_heads == 0)
        
        self.depth = self.d_model // self.num_heads
        
        self.wq = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))
        self.wk = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))
        self.wv = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))
        
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        
        x: in shape of (batch_size, seq_max_length, vocab_size)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        v = self.wv(v)
        k = self.wk(k)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    
    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
        output, attention_weights
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
    
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)
    
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
        output = tf.matmul(attention_weights, v)
    
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        out_final = self.layernorm2(out1 + out2)
        
        return out_final


class PositionalEncoding(tf.keras.layers.Layer):
    
    def __init__(self, max_length, max_depth, min_rate=1/10000, dtype=tf.float32, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        if max_depth % 2 != 0:
            warnings.warn("max_depth should be even, max_depth is incremented!")
            max_depth += 1

        pos = np.arange(max_length)
        i = np.arange(max_depth // 2)
        pos, i = np.meshgrid(pos, i)  # build the meshgrid of both pos and i
        embedding = np.empty((max_length, max_depth), )
        embedding[:, ::2] = np.sin(pos * min_rate**(2 * i / max_depth)).T
        embedding[:, 1::2] = np.cos(pos * min_rate**(2 * i / max_depth)).T
        # new axis is added for batches dimension
        # as we would like to broadcast positional embedding for all instance in
        # the batch
        self.positional_embedding = tf.constant(embedding[np.newaxis, ...], dtype=dtype)
  
    def call(self, x, seq_length=None):
        """
        :param X: the computed X of the embedding layer
        :param seq_length: The maximum length of the given input, if None then you need to make sure that
                        `max_length` = input's maximum sequence length otherwise, it is not broadcastable.
        """
        return x + self.positional_embedding[:, :seq_length, :]
    

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, N, d_model, num_heads, dff, vocab_size, max_positional_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.N = N
        self.pos_encoding = PositionalEncoding(max_positional_encoding, d_model)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(N)]
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # normalizing X before adding it to the positional encoding.
        x = self.pos_encoding(x, seq_length=seq_len)
        
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.encoder_layers[i](x, training, mask)
        
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = point_wise_feed_forward_network(d_model, num_heads)
        
    def call(self, x, enc_outputs, training, look_ahead_mask, padded_mask):
        
        self_attn, attn_w_block1 = self.mha1(x, x, x, look_ahead_mask)
        self_attn = self.dropout1(self_attn, training=training)
        out1 = self.layerNorm1(self_attn + x)
        
        attn, attn_w_block2 = self.mha2(enc_outputs, enc_outputs, out1, padded_mask)
        attn = self.dropout2(attn, training=training)
        out2 = self.layerNorm2(attn + out1)
        
        out3 = self.ffn(out2)
        out3 = self.dropout3(out3, training=training)
        out_final = self.layerNorm3(out3 + out2)
        
        return out_final, attn_w_block1, attn_w_block2


class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, N, d_model, num_heads, dff, vocab_size, max_positional_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.d_model = d_model
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_positional_encoding, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(N)]
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = dict()
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x, seq_length=seq_len)
    
        x = self.dropout(x, training=training)
        
        for i in range(self.N):
            x, att1, att2 = self.decoder_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = att2
        
        return x, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dff, activation='relu')),  # (batch_size, seq_len, dff)
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_model))  # (batch_size, seq_len, d_model)
    ])


class Transformer(tf.keras.Model):
    
    def __init__(self, N, d_model, num_heads, dff, input_vocab_size, target_vocab_size, 
                 input_max_positional_encoding, target_max_positional_encoding, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(N, d_model, num_heads, dff, input_vocab_size, input_max_positional_encoding, rate)
    
        self.decoder = Decoder(N, d_model, num_heads, dff, target_vocab_size, target_max_positional_encoding, rate)
        
        self.dense = tf.keras.layers.Dense(target_vocab_size, activation="softmax")
        
    def call(self, X, training=None, **kwargs):
        enc_x, dec_x = X
        # create the masks
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(enc_x, dec_x)
        # run encoder
        enc_y = self.encoder(enc_x, training, enc_padding_mask)
        # run decoder
        dec_y, atten_weights = self.decoder(dec_x, enc_y, training, look_ahead_mask, dec_padding_mask)
        # predict
        y = self.dense(dec_y)
        return y, atten_weights
    
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        upper_triangle_zeros = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        lower_traingle_diagonal_zeros = 1 - upper_triangle_zeros
        return lower_traingle_diagonal_zeros

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    
class TransformerV2(Transformer):
    
    def __init__(self, *args, **kwargs):
        super(TransformerV2, self).__init__(*args, **kwargs)
    
    def call(self, X, training=None, **kwargs):
        enc_x, dec_x = X
        # create the masks
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(enc_x, dec_x)
        # run encoder
        enc_y = self.encoder(enc_x, training, enc_padding_mask)
        # run decoder
        dec_y, _ = self.decoder(dec_x, enc_y, training, look_ahead_mask, dec_padding_mask)
        # predict
        y = self.dense(dec_y)
        return y


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}
        