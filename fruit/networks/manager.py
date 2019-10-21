import tensorflow as tf
import tflearn
from tensorflow.python.ops.nn import rnn_cell as _rnn_cell


# Minimal implementation of layer manager
class LayerManager(object):
    def __init__(self):
        self.num_of_inputs = 0
        self.num_of_hidden_layers = 0
        self.num_of_outputs = 0
        self.inputs = []
        self.hiddens = []
        self.outputs = []
        super().__init__()

    def create_input(self, type, shape, name=None):
        input = tf.placeholder(type, shape=shape, name=name)
        self.inputs.append(input)
        self.num_of_inputs += 1
        return input

    def create_fully_connected_layer(self, input_data, num_of_units, activation_fn='linear', scope = None):
        layer = tflearn.fully_connected(input_data, num_of_units, activation=activation_fn, scope=scope)
        self.hiddens.append(layer)
        self.num_of_hidden_layers += 1
        return layer

    def create_conv_layer(self, input_data, num_of_filters, filter_size, strides,
                          activation_fn='relu', padding='valid', permutation=None, scope=None):
        if permutation is not None:
            input_data = tf.transpose(input_data, permutation)
        conv = tflearn.conv_2d(input_data, nb_filter=num_of_filters, filter_size=filter_size, strides=strides,
                               activation=activation_fn, padding=padding, scope=scope)
        self.hiddens.append(conv)
        self.num_of_hidden_layers += 1
        return conv

    def create_output(self, input_data, output_size, activation_fn='linear', scope=None):
        output = tflearn.fully_connected(input_data, output_size, activation=activation_fn, scope=scope)
        self.outputs.append(output)
        self.num_of_outputs += 1
        return output

    # input_data must be [batch_size, data_size]
    def create_basic_lstm_layer(self, input_data, input_size, num_of_units, scope=None):

        batch_size = tf.shape(input_data)[0]

        input_data_reshape = tf.reshape(input_data, [1, -1, input_size])

        lstm_layer = tflearn.BasicLSTMCell(num_units=num_of_units, state_is_tuple=True)
        lstm_state_size = tuple([[1, x] for x in lstm_layer.state_size])

        initial_lstm_state = _rnn_cell.LSTMStateTuple(
            tf.placeholder(tf.float32, shape=lstm_state_size[0], name='initial_lstm_state1'),
            tf.placeholder(tf.float32, shape=lstm_state_size[1], name='initial_lstm_state2'))

        sequence_length = tf.reshape(batch_size, [1])

        lstm_output, new_lstm_state = tf.nn.dynamic_rnn(lstm_layer, input_data_reshape,
                                                        initial_state=initial_lstm_state,
                                                        sequence_length=sequence_length,
                                                        time_major=False, scope=scope)

        lstm_output_reshape = tf.reshape(lstm_output, [-1, num_of_units])

        self.hiddens.append(lstm_output_reshape)

        self.num_of_hidden_layers += 1

        return lstm_output_reshape, initial_lstm_state, new_lstm_state

    def get_num_of_inputs(self):
        return self.num_of_inputs

    def get_num_of_hidden_layers(self):
        return self.num_of_hidden_layers

    def get_num_of_outputs(self):
        return self.num_of_outputs

    def get_inputs(self):
        return self.inputs

    def get_hidden_layers(self):
        return self.hiddens

    def get_outputs(self):
        return self.outputs



