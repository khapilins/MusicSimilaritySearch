import tensorflow as tf


class ConvModel(object):
    '''Model, wich applies to input signal filterbank of horizontal convolutions,
    then filterbank(s) of vertical (dilated) convolutions, and feeds the result
    to bidirectional GRU'''

    def __init__(self,
                 kernel_sizes=[10, 8, 4, 4, 4],
                 strides=[5, 4, 2, 2, 2],
                 dilations=[1, 1, 1, 1, 1],
                 filters=8,
                 conv_keep_prob=0.7,
                 use_batch_norm=True,
                 proj_before_gru=8,
                 num_gru_units=16,
                 postproc_filters=16,
                 out_classes=10,
                 use_state=False,
                 dropout_rate=0.5,
                 is_training=True):
        '''Args:
            kernel_sizes -> list of int - kernel_sizes for 2D convolutions
            strides -> list of int - strides of 2D convolutions
            dilations -> list of int - 2D conv dilation rates
            conv_keep_prob -> float - dropout strength on convolutions
            use_batch_norm -> bool - whether to use batch norm
            proj_before_gru -> int - number of linear projection units
                before GRU layer, will be used as zs in CPC
            num_gru_units -> int - number of GRU units
            out_classes -> int - number of classes to predict
            use_state -> bool - whether to use state of rnn for prediction
            dropout_rate -> float - dropout on 2 last FC channels
            is_training -> bool or bool tensor - training phase'''
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilations = dilations
        self.filters = filters
        assert len(kernel_sizes) == len(dilations) == len(strides)
        self.conv_keep_prob = conv_keep_prob
        self.use_batch_norm = use_batch_norm,
        self.proj_before_gru = proj_before_gru
        self.num_gru_units = num_gru_units
        self.postproc_filters = postproc_filters
        self.out_classes = out_classes
        self.use_state = use_state
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.layers = []

    def create_model(self, input_tensor):
        # Just hack to avoid explicit zero-mean 1-std normalization
        if self.use_batch_norm:
            input_tensor = tf.layers.batch_normalization(
                input_tensor,
                training=self.is_training)
        out = tf.squeeze(input_tensor, [-2])

        with tf.variable_scope('conv_stack', reuse=tf.AUTO_REUSE):
            for i, (k_size, stride, dilation) in enumerate(zip(
                    self.kernel_sizes,
                    self.strides,
                    self.dilations)):
                conved = tf.layers.conv1d(out,
                                          self.filters,
                                          k_size,
                                          strides=stride,
                                          padding='valid',
                                          dilation_rate=dilation,
                                          activation=None,
                                          name='conv_stack_{}'.format(i))
                if self.use_batch_norm:
                    conved = tf.layers.batch_normalization(
                        conved,
                        training=self.is_training)
                conved = tf.nn.relu(conved)
                if self.conv_keep_prob < 1:
                    conved = tf.layers.dropout(conved,
                                               rate=self.conv_keep_prob,
                                               training=self.is_training)
                #conved += out
                self.layers.append(conved)
                out = conved

        with tf.variable_scope('linear_proj', reuse=tf.AUTO_REUSE):
            out = tf.layers.conv1d(out,
                                   self.proj_before_gru,
                                   1,
                                   name='linear_proj')
            self.layers.append(out)
            self.zs = out

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
