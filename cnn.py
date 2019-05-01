import tensorflow as tf
from flags import *

class CNN:
    def __init__(
      self, name, sequence_length, num_classes,
        filter_sizes, num_filters, init_emb, embedding_size, dropout_prob):
        # Placeholders for input, output and dropout
        self.name = name
        self.filters = []
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        with tf.device("/cpu:0"):
            self.emb = tf.Variable(init_emb, dtype=tf.float32, name="{}/emb".format(name), trainable=True)
        self.embedding_size = embedding_size
        self.dropout_prob = dropout_prob
        self.initialized = False


    def network(self, input_x, avg_pool = False):
        # Keeping track of l2 regularization loss (optional)
        with tf.variable_scope(self.name, reuse=self.initialized):
            self.initialized = True
            self.l2_loss = tf.constant(0.0)
            # Embedding layer
            with tf.name_scope("embedding"):
                raw_embedded_chars = tf.nn.embedding_lookup(self.emb, input_x)
                emb_flat = tf.reshape(raw_embedded_chars, [-1, self.embedding_size])
                self.embedded_chars = tf.reshape(emb_flat, [-1, self.sequence_length, self.embedding_size])
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            arg_list = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    self.filters.append((W,b))

                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    if avg_pool:
                        pooled = tf.nn.avg_pool(
                            h,
                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                    else:
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                    arg = tf.argmax(h, axis=1)
                    arg_list.append(arg)
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.pooledArg = tf.concat(arg_list, 2)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_prob)

            # Final (unnormalized) scores and predictions
            with tf.variable_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    )
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
                self.dense_W = W
                self.dense_b = b
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                self.logit = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") # [batch, num_class]
                self.scores = tf.nn.softmax(self.logit)
                return self.logit