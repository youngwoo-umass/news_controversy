import tensorflow as tf
import numpy as np
from load_data import *
import os
from neural_model import ADReaction
from flags import *


def average(l):
    return sum(l) / len(l)


def get_batches(data, batch_size):
    X, Y = data
    # data is fully numpy array here
    step_size = int((len(Y) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        x = []
        y = []
        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(Y):
                break
            x.append(X[idx])
            y.append(Y[idx])
        if len(y) > 0:
            new_data.append((np.array(x),np.array(y)))
    return new_data


class CommentClassifier:
    def __init__(self):
        self.prior = 0.5 # Expected controversy portion in the dats

        self.ini_emb = load_init_embbedding()
        self.prior = 0.5
        self.disagree_model = ADReaction(self.prior, self.ini_emb)
        self.sess = self.init_sess()
        self.lr = 1e-3
        self.reg_lambda = 1e-1
        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.num_filters = 128

    @staticmethod
    def init_sess():
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True

        return tf.Session(config=config)


    def load_disagree_model(self):
        self.sess.run(tf.global_variables_initializer())
        save_dir = os.path.join(data_path, 'agree')
        path = os.path.join(save_dir, "model-36570")
        print("Loading model from : {}".format(path))
        def condition(v):
            if v.name.split('/')[0] == 'agree':
                return True
            return False

        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if condition(v)]
        saver = tf.train.Saver(variables_to_restore, max_to_keep=1)
        saver.restore(self.sess, path)

    @staticmethod
    def reshape_comment(comment):
        comment = comment[:FLAGS.comment_count, :FLAGS.comment_length]
        pad_size = FLAGS.comment_count - comment.shape[0]
        return np.pad(comment, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)

    def get_score(self, input_reaction):
        scores, = self.sess.run([self.disagree_model.score, ],
                                  feed_dict={
                                      self.disagree_model.input_comment : input_reaction,
                                      self.disagree_model.dropout_keep_prob: 0.5,
                                  })
        return scores[:,1]

    def set_threshold_predict(self, reaction_list):
        # reaction_list: List[ discussion_id, url, np.array#[count, max_length]
        # Train Goal : Learn [b] to get 50%(prior) controversy
        batch_size = 10

        # batch_size * comment_count * comment_length * embedding_size * filter
        # 10 * 100 * 50 * 100 * 60

        def get_batches_w_pad(reactions, batch_size):
            l = len(reactions)
            step_size = int((l + batch_size - 1) / batch_size)
            new_data = []

            def pad_if_need(reaction):
                comment = reaction[2]
                comment = comment[:FLAGS.comment_count, :FLAGS.comment_length]
                pad_size = FLAGS.comment_count - comment.shape[0]
                return np.pad(comment, [(0, pad_size), (0, 0)], mode='constant', constant_values=0)

            for step in range(step_size):
                X = []
                for i in range(batch_size):
                    idx = step * batch_size + i
                    if idx >= l:
                        X.append(np.zeros([FLAGS.comment_count, FLAGS.comment_length]))
                    else:
                        X.append(pad_if_need(reactions[idx]))
                new_data.append(X)
            return new_data

        #  predict all scores
        batches = get_batches_w_pad(reaction_list, batch_size)
        all_score = []

        sample_size = len(batches)
        print("Evaluating Scores : total {} steps".format(sample_size))
        for batch in batches[:sample_size]:
            scores = self.get_score(batch)
            all_score.append(scores)
        all_score = np.concatenate(all_score)
        cut_idx = int(self.prior * len(all_score))
        threshold = sorted(all_score, reverse=True)[cut_idx]
        print("Median controversy score : {}".format(threshold))
        # validate
        avg_list = []
        all_label = []
        for batch in batches:
            scores = self.get_score(batch)
            label = np.less(threshold, scores)
            all_label.append(label)
            c_count = np.count_nonzero(label)
            portion = c_count / len(label)
            avg_list.append(portion)
        all_label = np.concatenate(all_label)
        c_avg = average(avg_list)
        print("Portion of controversial docs : {}".format(c_avg))
        assert (self.prior - 0.05 < c_avg < self.prior + 0.05)
        label_dict = dict()
        for i, r in enumerate(reaction_list):
            article_id = reaction_list[i][0]
            label_dict[article_id] = all_label[i]

        save_reaction_label_cache(label_dict, "1")
        return label_dict

    def train_disagree_classifier(self, data):
        print("Train_disagree_classifier ENTRY")
        voca = load_voca()
        comments = load_agree_data()
        X, Y = format_comment(comments, voca, FLAGS.comment_length)
        epoch = 1
        batch_size = 100
        batches = get_batches(data, batch_size)


        loss = self.disagree_model.agree_loss + self.reg_lambda * self.disagree_model.get_l2_loss()
        train_op = self.get_train_op(loss)
        self.sess.run(tf.global_variables_initializer())


        def train_step(batch, i):
            def pad_reformat(batch):  # Pad the comments to match the shape

                text, y = batch
                unit = self.disagree_model.comment_count
                init = len(y)
                remn = init % unit
                pad_len = unit - remn if remn > 0 else 0
                text_pad = np.pad(text, [(0, pad_len), (0, 0)], mode='constant', constant_values=0)
                y_pad = np.concatenate([y, np.zeros(pad_len)], axis=0)
                return (np.reshape(text_pad, [-1, unit, self.disagree_model.comment_length]),
                        np.reshape(y_pad, [-1, unit]))

            text, y = pad_reformat(batch)
            loss_v, acc, _ = self.sess.run([loss, self.disagree_model.agree_acc, train_op],
                                         feed_dict={
                                             self.disagree_model.input_comment: text,
                                             self.disagree_model.input_comment_y: y,
                                             self.disagree_model.dropout_keep_prob: 0.5,
                                         })
            print("Step {} : Loss={} Acc={}".format(i, loss_v, acc))
            return loss, acc

        def epoch_runner(batches, step_fn, dev_fn=None, valid_freq=1000):
            l_loss = []
            l_acc = []
            valid_stop = 0
            np.random.shuffle(batches)
            for i, batch in enumerate(batches):
                if dev_fn is not None:
                    if valid_freq == valid_stop:
                        dev_fn()
                        valid_stop = 0
                    valid_stop += 1

                loss, acc = step_fn(batch, i)
                l_acc.append(acc)
                l_loss.append(loss)
            return average(l_loss), average(l_acc)

        def save_agree():
            id = 'agree_1'
            os.mkdir(os.path.join(data_path, 'runs'))
            save_dir = os.path.join(data_path, 'runs', id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            path = os.path.join(save_dir, "model")
            self.saver.save(self.sess, path, global_step=self.global_step)

        for i in range(epoch):
            loss, acc = epoch_runner(batches, train_step)
            print("Epoch {} Loss={} Acc={}".format(i, loss, acc))
            save_agree()

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        return optimizer.apply_gradients(grads_and_vars,
                                         global_step=self.global_step)
