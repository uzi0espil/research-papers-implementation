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
        if not self.bidirectional:
            output, hidden_states, cell_states = self.lstm(x)
        else:
            output, f_hidden_states, f_cell_states, b_hidden_states, b_cell_states = self.lstm(x)
            hidden_states = (f_hidden_states, b_hidden_states)
            cell_states = (f_cell_states, b_cell_states)
        return output, (hidden_states, cell_states)
    

class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))
        self.W2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))
        self.V = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))  # context Vector
        
    def call(self, query, values):
        """
        query is the previous hidden state of the output, that will use to query which part of the encoder inputs (values) to concentrate on.
        """
        
        query_with_time_axis = tf.expand_dims(query, axis=1)
        
        # score has shape of (batch_size, max_length, 1)  # for each batch, for each time step we compute a similarity score.
        # for each batch, we add the current weighted hidden state to each weighted timestep in the input.
        x = self.W1(query_with_time_axis) + self.W2(values)
        score = self.V(tf.nn.tanh(x)) # apply tanh and get a single value for each batch.
        
        # then compute the attention distribution
        attention_weights = tf.nn.softmax(score, axis=1)  # axis 1 to apply for each time step for each batch.
        
        # compute the context vector (which is the sum of multiplication of each step with its corresponding attiontion weight)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    

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
        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")
        
        # attentions layer
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, encoder_output):
        """
        :param x: is the other language input.
        :param hidden: the encoder's hidden states
        :encoder_output: the encoder's output
        """
        if self.bidirectional:
            hidden, cell = hidden
            # based on the paper, we concatenate the forward and backward hidden together.
            query = tf.concat(hidden, axis=-1)
            hidden = [hidden[0], cell[0], hidden[1], cell[1]]
        else:
            query = hidden[0]
            
        
        context_vector, attention_weights = self.attention(query, encoder_output)
        
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        if not self.bidirectional:
            output, hidden_states, cell_states = self.lstm(x, initial_state=hidden)
        else:
            output, f_hidden_states, f_cell_states, b_hidden_states, b_cell_states = self.lstm(x)
            hidden_states = (f_hidden_states, b_hidden_states)
            cell_states = (f_cell_states, b_cell_states)
        
        # Flattening the results
        output = tf.reshape(output, shape=(-1, output.shape[2]))
        
        x = self.fc(output)  # it doesn't do softmax as the loss function will deal with logits and not probabilities.
        
        return x, (hidden_states, cell_states), attention_weights
