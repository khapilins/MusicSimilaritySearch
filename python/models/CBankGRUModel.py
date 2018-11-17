import tensorflow as tf


class CBankGRUModel(object):
    '''Model, which applies to input signal filterbank of horizontal convolutions,
    then filterbank(s) of vertical (dilated) convolutions, and feeds the result
    to bidirectional GRU'''

    def __init__(self,
                 horizontal_kernel_sizes=[2**i for i in range(5)],
                 horizontal_dilations=[1]*5,
                 horizontal_filters=16,
                 vertical_kernel_size=[4, 4, 4],
                 vertical_dilations=[1, 4, 16],
                 vertical_filters=16,
                 conv_keep_prob=0.7,
                 use_batch_norm=True,
                 num_gru_units=128,  # note that it's for 1 direction
                 postproc_filters=128,
                 out_classes=10,
                 use_state=False,
                 dropout_rate=0.5,
                 unidirectional=False,
                 is_training=True):
        '''Args:
            horizontal_kernel_sizes -> list of int - kernel_size for 1D
                frequency-band-wise convolution
            horizontal_dilations -> list of int - 1D conv dilation rates
            vertical_kernel_size-> list of int - kernel_size for 1D
                time-wise convolution
            vertical_dilations -> list of int - 1D conv dilation rates
            conv_keep_prob -> float - dropout strength on convolutions
            use_batch_norm -> bool - whether to use batch norm
            num_gru_units -> int - number of GRU units in a single direction
            out_classes -> int - number of classes to predict
            use_state -> bool - whether to use state of rnn for prediction
            dropout_rate -> float - dropout on 2 last FC channels
            is_training -> bool or bool tensor - training phase'''
        self.horizontal_kernel_sizes = horizontal_kernel_sizes
        self.horizontal_dilations = horizontal_dilations
        self.horizontal_filters = horizontal_filters
        assert len(horizontal_dilations) == len(horizontal_kernel_sizes)
        self.vertical_kernel_size = vertical_kernel_size
        self.vertical_dilations = vertical_dilations
        self.vertical_filters = vertical_filters
        assert len(vertical_dilations) == len(vertical_kernel_size)
        self.conv_keep_prob = conv_keep_prob
        self.use_batch_norm = use_batch_norm
        self.num_gru_units = num_gru_units
        self.postproc_filters = postproc_filters
        self.out_classes = out_classes
        self.use_state = use_state
        self.dropout_rate = dropout_rate
        self.unidirectional = unidirectional
        self.is_training = is_training

    def create_model(self, input_tensor):
        # Just hack to avoid explicit zero-mean 1-std normalization
        if self.use_batch_norm:
            input_tensor = tf.layers.batch_normalization(
                input_tensor,
                training=self.is_training)
        out = input_tensor

        with tf.variable_scope('horizontal_stack', reuse=tf.AUTO_REUSE):
            out_proj = tf.layers.conv2d(
                input_tensor,
                self.horizontal_filters * len(self.horizontal_kernel_sizes),
                (1, 1),
                strides=(1, 1),
                padding='same',
                activation=None,
                name='hor_inp_proj')
            self.horizontal_input_proj = out_proj
            self.h_stack = []
            for i, (k_size, dilation) in enumerate(zip(
                    self.horizontal_kernel_sizes,
                    self.horizontal_dilations)):
                conved = tf.layers.conv2d(out,
                                          self.horizontal_filters,
                                          # Height 1, width from parameters
                                          (1, k_size),
                                          strides=(1, 1),
                                          padding='same',
                                          dilation_rate=(1, dilation),
                                          activation=None,
                                          name='h_stack_conv_{}'.format(i))
                if self.use_batch_norm:
                    conved = tf.layers.batch_normalization(
                        conved,
                        training=self.is_training)
                conved = tf.nn.relu(conved)
                if self.conv_keep_prob < 1:
                    conved = tf.layers.dropout(conved,
                                               rate=self.conv_keep_prob,
                                               training=self.is_training)
                self.h_stack.append(conved)
            self.h_stacked = tf.concat(self.h_stack, 3) + out_proj
            out = self.h_stacked

        with tf.variable_scope('vertical_stack', reuse=tf.AUTO_REUSE):
            out_proj = tf.layers.conv2d(
                out,
                self.vertical_filters * len(self.vertical_kernel_size),
                (1, 1),
                strides=(1, 1),
                padding='same',
                activation=None,
                name='ver_inp_proj')
            self.vertical_out_proj = out_proj
            self.v_stack = []
            for i, (k_size, dilation) in enumerate(zip(
                    self.vertical_kernel_size,
                    self.vertical_dilations)):
                conved = tf.layers.conv2d(out,
                                          self.vertical_filters,
                                          (k_size, 1),
                                          strides=(1, 1),
                                          padding='same',
                                          dilation_rate=(dilation, 1),
                                          activation=None,
                                          name='v_stack_conv_{}'.format(i))
                if self.use_batch_norm:
                    conved = tf.layers.batch_normalization(
                        conved,
                        training=self.is_training)
                conved = tf.nn.relu(conved)
                if self.conv_keep_prob < 1:
                    conved = tf.layers.dropout(conved,
                                               self.conv_keep_prob,
                                               training=self.is_training)
                self.v_stack.append(conved)
            self.v_stacked = tf.concat(self.v_stack, 3) + out_proj
            out = self.v_stacked
        self.zs = tf.squeeze(out, axis=[-2])
        with tf.variable_scope('BiGRU', reuse=tf.AUTO_REUSE):
            cell_fwd = tf.contrib.rnn.GRUCell(self.num_gru_units)
            cell_bwd = tf.contrib.rnn.GRUCell(self.num_gru_units)
            out = tf.squeeze(out, axis=[-2])
            output, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fwd,
                cell_bwd,
                out,
                swap_memory=True,
                dtype=tf.float32)
            self.gru_out, self.state = (tf.concat(output, 2),
                                        tf.concat(state, 1))
            if self.use_state:
                proj_inp = self.state[:, None, :]
            else:
                proj_inp = self.gru_out
        if self.unidirectional:
            with tf.variable_scope('UniGRU', reuse=tf.AUTO_REUSE):
                cell = tf.contrib.rnn.GRUCell(self.num_gru_units)
                output, state = tf.nn.dynamic_rnn(cell,
                                                  self.zs,
                                                  swap_memory=True,
                                                  dtype=tf.float32)
                self.gru_out, self.state = output, state
                if self.use_state:
                    proj_inp = self.state[:, None, :]
                else:
                    proj_inp = self.gru_out

                # For CPC; this is our auto-regressive output
                self.cts = self.gru_out

        with tf.variable_scope('postproc', reuse=tf.AUTO_REUSE):
            proj_inp = tf.layers.dropout(proj_inp,
                                         self.dropout_rate,
                                         training=self.is_training)
            output = tf.layers.dense(proj_inp,
                                     self.postproc_filters,
                                     activation=tf.nn.relu,
                                     name='postproc1')
            # for classification embeddings
            self.dense_out = output
            output = tf.layers.dropout(output,
                                       self.dropout_rate,
                                       training=self.is_training)
            output = tf.layers.dense(output,
                                     self.out_classes,
                                     activation=tf.nn.relu,
                                     name='postproc2')
            output = tf.reduce_mean(output, axis=[1])

            self.prediction = output
            return output
