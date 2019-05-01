from LM import LM
import numpy as np
from comment_classifier_1 import CommentClassifier
from load_data import *
from flags import *

def split_LM_docs(articles, id2label):
    c_docs = []
    nc_docs = []
    for shord_id, document in articles:
        if shord_id not in id2label:
            continue
        if id2label[shord_id]:
            c_docs.append(document)
        else:
            nc_docs.append(document)
    return c_docs, nc_docs


def doc_eval(classifier, eval_data):
    dev_x, x_comment, dev_y = eval_data
    valid_size = len(dev_y)
    suc_count = 0
    pred_list = []
    for i in range(valid_size):
        pred = classifier.predict(dev_x[i])
        if pred == dev_y[i]:
            suc_count += 1
        pred_list.append(pred)

    acc = suc_count / valid_size
    return acc


def train_article_classifier1(articles, reaction_label, eval_data):
    # count word occurence in C/NC
    c_docs, nc_docs = split_LM_docs(articles, reaction_label)
    LM_classifier = LM.train(c_docs, nc_docs)
    acc = doc_eval(LM_classifier, eval_data)
    print("Eval Acc:\t{}".format(acc))
    return LM_classifier

def train_doc_by_LM(reaction_list, articles, LM, eval_data):
    label_dict = dict()
    id_list = []
    input_list = []
    for discussion_id, _, comments in reaction_list:
        input = np.reshape(comments, [-1])
        input_list.append(input)
        id_list.append(discussion_id)
    print("Predicting..")
    y_list = LM.predict_parallel(input_list)

    for id, y in zip(id_list, y_list):
        label_dict[id] = y

    c_docs, nc_docs = split_LM_docs(articles, label_dict)
    print("Training..")
    doc_LM = LM.train(c_docs, nc_docs)

    acc = doc_eval(doc_LM, eval_data)
    print("Eval Acc:\t{}".format(acc))
    return doc_LM


def train_comment_LM(reaction_list, articles, LM, eval_data):
    c_docs = []
    nc_docs = []
    ids, document = zip(*articles)
    labels = LM.predict_parallel(document)
    label_dict = dict(zip(ids, labels))
    for discussion_id, _, comments in reaction_list:
        if discussion_id in label_dict:
            if label_dict[discussion_id]:
                c_docs += comments.tolist()
            else:
                nc_docs += comments.tolist()

    print("Training")
    LM_classifier = LM.train(c_docs, nc_docs)
    x_doc, x_comment, y = eval_data

    suc_count = 0
    dev_x = []
    y_s = []
    for i in range(len(y)):
        input = np.reshape(x_comment[i], [-1])
        dev_x.append(input)
        y_pred = LM_classifier.predict(input)
        if y[i] == y_pred:
            suc_count += 1
        y_s.append(y[i])

    valid_size = len(y)
    acc = suc_count / valid_size
    scores = [LM_classifier.cont_score(d) for d in dev_x]

    print("Eval Acc:\t{}".format(acc))
    return LM_classifier



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


def predict_by_disagreement(reaction_list, use_trained=True):
    if not use_trained :
        c = CommentClassifier()
        comments = load_agree_data()
        X, Y = format_comment(comments, load_voca(), FLAGS.comment_length)
        c.train_disagree_classifier((X,Y))
        return c.set_threshold_predict(reaction_list)
    else:
        c = CommentClassifier()
        c.load_disagree_model()
        return c.set_threshold_predict(reaction_list)