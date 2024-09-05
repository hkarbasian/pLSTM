import tensorflow as tf

#---------------------------------------------------------------------
# customized pLSTM
#---------------------------------------------------------------------
class Customized_LSTM(tf.keras.layers.Layer):
    def __init__(self, units, num_outputs):
        super().__init__()
        self.units = units
        self.num_outputs = num_outputs
        self.activation = tf.keras.activations.get('tanh')
        self.recurrent_activation = tf.keras.activations.get('sigmoid')

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # learning weights and bias
        self.U = tf.Variable(tf.zeros(shape=(input_dim, 4*self.units)), name='U')
        self.W = tf.Variable(tf.zeros(shape=(self.units, 4*self.units)), name='W')
        self.b = tf.Variable(tf.zeros(shape=(4*self.units,)), name='bias')

        """self.U = tf.keras.initializers.get('glorot_uniform')(shape=(input_dim, 4*self.units))
        self.W = tf.keras.initializers.get('orthogonal')(shape=(self.units, 4*self.units))

        self.b = tf.keras.backend.concatenate(
            [
                tf.zeros(shape=(self.units,)), 
                tf.zeros(shape=(self.units,)), 
                tf.zeros(shape=(2*self.units,))
            ] 
            )"""
        
    def call(self, inputs):

        # initialized hidden states
        h_t = tf.zeros((tf.shape(inputs)[0], self.units))
        c_t = tf.zeros((tf.shape(inputs)[0], self.units))

        outputs = []
        num_inputs = inputs.shape[1]
        tr  = num_inputs - self.num_outputs
        for t in range(num_inputs):
            x_t = inputs[:, t, :]
            z = tf.matmul(x_t, self.U) + tf.matmul(h_t, self.W) + self.b
            z0, z1, z2, z3 = tf.split(z, 4, axis=1)

            # gates
            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c_t = f * c_t + i * self.activation(z2)
            o = self.recurrent_activation(z3)
            h_t = o * self.activation(c_t)

            # number of outputs to be stored
            if t >= tr:
                outputs.append(h_t)
        
        outputs = tf.stack(outputs, axis=1)
        return outputs

#---------------------------------------------------------------------
# ghost-pLSTM
#---------------------------------------------------------------------
class Parametric_LSTM(tf.keras.layers.Layer):
    def __init__(self, units, num_outputs):
        super().__init__()
        self.units = units
        self.num_outputs = num_outputs
        self.activation = tf.keras.activations.get('tanh')
        self.recurrent_activation = tf.keras.activations.get('sigmoid')

    def build(self, input_shape):

        input_dim = input_shape[0][-1]
        param_dim = input_shape[1][-1]

        # learning weights and bias
        self.U = tf.Variable(tf.zeros(shape=(input_dim, 4*self.units)), name='U')
        self.W = tf.Variable(tf.zeros(shape=(self.units, 4*self.units)), name='W')
        self.S = tf.Variable(tf.zeros(shape=(param_dim, 4*self.units)), name='W')
        self.b = tf.Variable(tf.zeros(shape=(4*self.units,)), name='bias')

    def call(self, inputs):

        X_t = inputs[0]
        S_t = inputs[1]

        # initialized hidden states
        h_t = tf.zeros((tf.shape(X_t)[0], self.units))
        c_t = tf.zeros((tf.shape(X_t)[0], self.units))

        outputs = []
        num_inputs = X_t.shape[1]
        tr  = num_inputs - self.num_outputs
        for t in range(num_inputs):
            x_t = X_t[:, t, :]
            s_t = S_t[:, t, :]
            z = tf.matmul(x_t, self.U) + tf.matmul(h_t, self.W)  + tf.matmul(s_t, self.S) + self.b
            z0, z1, z2, z3 = tf.split(z, 4, axis=1)

            # gates
            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c_t = f * c_t + i * self.activation(z2)
            o = self.recurrent_activation(z3)
            h_t = o * self.activation(c_t)

            # number of outputs to be stored
            if t >= tr:
                outputs.append(h_t)
        
        outputs = tf.stack(outputs, axis=1)
        return outputs

#---------------------------------------------------------------------
# 
#---------------------------------------------------------------------