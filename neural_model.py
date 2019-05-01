import numpy as np
import tensorflow as tf

from flags import *
from cnn import CNN



def accuracy(logits, y, axis=1):
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, axis=axis),
                         tf.cast(y, tf.int64)), tf.float32))


# ADReaction : Agreement/Disagreement in reaction(comments). We actually use only disagreement expression as signal.
class ADReaction():
    def __init__(self, prior, init_emb):
        self.comment_length = FLAGS.comment_length
        self.comment_count = FLAGS.comment_count
        self.embedding_size = FLAGS.embedding_size
        self.prior = prior
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_comment = tf.placeholder(tf.int32,
                                            shape=[None, self.comment_count, self.comment_length],
                                            name="input_reaction")

        self.input_comment_y = tf.placeholder(tf.int32,
                                              shape=[None, self.comment_count],
                                              name="input_y_comment")  # agree label for comments

        self.input_y = tf.placeholder(tf.int32,
                                              shape=[None, ],
                                              name="input_y")  # Controversy Label


        self.cnn = CNN("agree",
                       sequence_length=self.comment_length,
                       num_classes=3,
                       filter_sizes=[1, 2, 3],
                       num_filters=64,
                       init_emb=init_emb,
                       embedding_size=self.embedding_size,
                       dropout_prob=self.dropout_keep_prob
                       )
        self.score = self.controversy(self.input_comment)
        self.acc = accuracy(self.score, self.input_y, axis=1)
        self.agree_logit = self.predict_2d(self.input_comment)
        self.agree_acc = accuracy(self.agree_logit, self.input_comment_y, axis=2)
        self.agree_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.agree_logit,
            labels=self.input_comment_y))

    def predict(self, comments):
        # comments : [None, 30]
        logits = self.cnn.network(comments)
        return logits  # logit : [None, 3]

    def predict_2d(self, comments):
        flat_comments = tf.reshape(comments, [-1, self.comment_length])
        logits = self.predict(flat_comments)
        formatted_logit = tf.reshape(logits, [-1, self.comment_count, 3])
        return formatted_logit

    def controversy(self, comments):
        formatted_logit = self.predict_2d(comments)
        avg = tf.reduce_sum(tf.nn.softmax(formatted_logit), axis=1)
        ad_weights = [[0,0], [0,0], [0,1]] # Only using disagreement as signal
        self.W = tf.Variable(ad_weights, trainable=False, dtype=tf.float32, name="ad_W")
        self.b = tf.Variable([0, 0], dtype=tf.float32, name="ad_b")
        score = tf.nn.xw_plus_b(avg, self.W, self.b)  # [None, 2]
        return score

    def get_l2_loss(self):
        return self.cnn.l2_loss


    def agree_2d(self, comments):
        flat_comments = tf.reshape(comments, [-1, self.comment_length])
        logits = self.agree(flat_comments)
        formatted_logit = tf.reshape(logits, [-1, self.comment_count, 3])
        return formatted_logit

