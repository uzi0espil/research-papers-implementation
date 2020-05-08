import tensorflow as tf


class CNNEncoder(tf.keras.Model):
    
    def __init__(self, units):
        super(CNNEncoder, self).__init__()
        # load Inception model and freeze the upper layers
        self.base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
        for layer in self.base_model.layers:
            layer.trainable = False
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation="relu"))
        
    def call(self, X):
        X = self.base_model(X)
        # merged the width and hight of feature map into a vector. so the reshape will convert (None, 8, 8, 2048) to (None, 64, 2048)
        X = tf.reshape(X, (X.shape[0], -1, X.shape[3]))
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
    
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.dot = tf.keras.layers.Dot((2, 2))
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation='tanh'))
        
    def call(self, dec_outputs, enc_outputs):
        scores = self.dot([dec_outputs, enc_outputs])
        
        attention_weights = tf.nn.softmax(scores)
        
        context_vector = tf.keras.layers.dot([attention_weights, enc_outputs], axes=(2,1))
        
        y = tf.keras.layers.concatenate([context_vector, dec_outputs])
        y = self.fc(y)
        
        return y, attention_weights
    

class RNNDecoder(tf.keras.Model):
    
    attentions = ["additive", "multiplicative"]
    
    def __init__(self, vocab_size, embedding_dim, units, attention="additive"):
        super(RNNDecoder, self).__init__()
        if attention is not None and attention.lower() not in self.attentions:
            raise ValueError("attention value is not correct, should be either {} or None".format(attentions))
        elif attention.lower() == "additive":
            self.attention = BahdanauAttention(units)
        elif attention.lower() == "multiplicative":
            self.attention = LuongAttention(units)
        else:
            self.attention = None
        
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(units, return_state=True, return_sequences=True)
        
        self.fc1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))
        self.fc2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation="softmax"))
        
    def call(self, x, hidden, enc_output):
        attention_weights = None
        x = self.embedding(x)
        
        if isinstance(self.attention, BahdanauAttention):
            # we only care about hidden states when computing the attention
            context_vector, attention_weights = self.attention(hidden[0], enc_output)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        x, h_states, c_states = self.lstm(x, initial_state=hidden)
        states = [h_states, c_states]
        
        if isinstance(self.attention, LuongAttention):
            x, attention_weights = self.attention(x, enc_output)
        
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(x)
        
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)
        
        return x, states, attention_weights
        
    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
    

class ImageCaption(tf.keras.Model):
    
    def __init__(self, target_vocab_size, vocab_size, embedding_dim, units):
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(target_vocab_size, vocab_size, embedding_dim, units, attention="multiplicative")
        
    def call(X, **kwargs):
        enc_input, dec_input = X['enc_input'], X['dec_input']
        hidden = self.decoder.reset_state(X.shape[0])
        
        enc_output = self.encoder(enc_input)
        decoder_output, attention = self.decoder(dec_input, hidden, enc_input)
        
        return decoder_output, attention