import tensorflow as tf


class CNNEncoder(tf.keras.layers.Layer):
    
    def __init__(self, units, inception_shape):
        super(CNNEncoder, self).__init__()
        # load Inception model and freeze the upper layers
        self.base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        self.reshape = tf.keras.layers.Reshape(inception_shape)
        for layer in self.base_model.layers:
            layer.trainable = False
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation="relu"))
        
    def call(self, X, training=None):
        X = self.base_model(X, training=training)
        # merged the width and hight of feature map into a vector. so the reshape will convert (None, 8, 8, 2048) to (None, 64, 2048)
        X = self.reshape(X)
        return self.fc(X)
    

class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))
        self.W2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))
        self.V = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        
    def call(self, query, values):
        
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        scores = tf.nn.tanh(self.W1(hidden_with_time_axis) + self.W2(values))
        
        attention_weights = tf.nn.softmax(self.V(scores), axis=1)
        
        context_values = attention_weights * values
        context_values = tf.reduce_sum(context_values, axis=1)
        
        return context_values, attention_weights
    

class LuongAttention(tf.keras.layers.Layer):
    
    def __init__(self, units, dropout_rate=None):
        super(LuongAttention, self).__init__()
        self.dot = tf.keras.layers.Dot((2, 2))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation='tanh'))
        
    def call(self, dec_outputs, enc_outputs, training):
        scores = self.dot([dec_outputs, enc_outputs])
        
        attention_weights = tf.nn.softmax(scores)
        
        # apply dropout if any.
        attention_weights = self.dropout(attention_weights, training=training)
        
        context_vector = tf.keras.layers.dot([attention_weights, enc_outputs], axes=(2,1))
        
        return context_vector, attention_weights
    

class RNNDecoder(tf.keras.layers.Layer):
    
    attentions = ["additive", "multiplicative"]
    
    def __init__(self, vocab_size, embedding_dim, units, attention="additive", 
                 mask=True, activation="linear", dropout_rate=None):
        super(RNNDecoder, self).__init__()
        if attention is not None and attention.lower() not in self.attentions:
            raise ValueError("attention value is not correct, should be either {} or None".format(attentions))
        elif attention.lower() == "additive":
            self.attention = BahdanauAttention(units)
        elif attention.lower() == "multiplicative":
            self.attention = LuongAttention(units, dropout_rate=dropout_rate)
        else:
            self.attention = None
        
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=mask)
        self.lstm = tf.keras.layers.LSTM(units, return_state=True, return_sequences=True)
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.fc1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation=activation))
        self.fc2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation="softmax"))
        
    def call(self, x, hidden, enc_output, training=None):
        attention_weights = None
        x = self.embedding(x)
        
        if isinstance(self.attention, BahdanauAttention):
            # we only care about hidden states when computing the attention
            context_vector, attention_weights = self.attention(hidden[0], enc_output)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        x, h_states, c_states = self.lstm(x)
        states = [h_states, c_states]
        
        if isinstance(self.attention, LuongAttention):
            context_vector, attention_weights = self.attention(x, enc_output, training)
            x = tf.keras.layers.concatenate([context_vector, x])
        
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(x)
        
        # dropout
        x = self.dropout(x)
        
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        
        return x, states, attention_weights
        
    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]


class ImageCaption(tf.keras.Model):
    """
    Model that encapsulate Encoder and Decoder layers. Supports only Luong Attention.
    """
    
    def __init__(self, inception_shape, target_vocab_size, embedding_dim, units, dropout_rate=None):
        super(ImageCaption, self).__init__()
        self.encoder = CNNEncoder(units, inception_shape)
        self.decoder = RNNDecoder(target_vocab_size, embedding_dim, units, 
                                  attention="multiplicative", activation="tanh",
                                  dropout_rate=dropout_rate)
        
    def call(self, X, training=None, **kwargs):
        enc_input, dec_input = X
        
        enc_output = self.encoder(enc_input, training=training)
        
        decoder_output, _, attention_weights = self.decoder(dec_input, None, enc_output, 
                                                            training=training)
        
        return decoder_output, attention_weights