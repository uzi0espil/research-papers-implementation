import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, bidirectional=False):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.bidirectional = bidirectional
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, 
                                         return_state=True, recurrent_initializer='glorot_uniform')
        if bidirectional:
            self.lstm = tf.keras.layers.Bidirectional(self.lstm)
        
    def call(self, x, **kwargs):
        x = self.embedding(x)
        x_and_states = self.lstm(x)
        return x_and_states[0], x_and_states[1:]
    

class LuongAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.dot = tf.keras.layers.Dot((2, 2))
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation='tanh'))
        
    def call(self, dec_outputs, enc_outputs):
        scores = self.dot([dec_outputs, enc_outputs])
        
        attention_weights = tf.nn.softmax(scores)
        
        context_vector = tf.keras.layers.dot([attention_weights, enc_outputs], axes=(2, 1))
        
        y = tf.keras.layers.concatenate([context_vector, dec_outputs])
        y = self.fc(y)
        
        return y, attention_weights
    

class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, dec_units, bidirectional=False):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.bidirectional = bidirectional
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(dec_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
        if bidirectional:
            self.lstm = tf.keras.layers.Bidirectional(self.lstm)
        
        # output layer
        self.fc1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dec_units, activation='relu'))
        self.fc2 = tf.keras.layers.Dense(vocab_size, activation="softmax")
        
        # attentions layer
        self.attention = LuongAttention(self.dec_units)
        
    def call(self, x, hidden, encoder_output):
        """
        :param x: is the other language input.
        :param hidden: the encoder's hidden states
        :encoder_output: the encoder's output
        """
        x = self.embedding(x)
        
        x_and_states = self.lstm(x, initial_state=hidden)
        x, states = x_and_states[0], x_and_states[1:]
        
        x, attention_weights = self.attention(x, encoder_output)
        
        x = self.fc1(x)
        
        # flatten x
        x = tf.reshape(x, (-1, x.shape[2]))
        
        output = self.fc2(x)
        
        return output, states, attention_weights
