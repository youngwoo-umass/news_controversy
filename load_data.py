import os
import pickle
import numpy as np
import re
data_path = "data"

def load_pickled_data(name):
    path = os.path.join(data_path, name)
    return pickle.load(open(path, "rb"))

def load_reactions():
    # return List[Discussion]
    # Discussion = (dicussion_id, web_url, np.array(code_comments))
    # code_comments has type of np.array and shape of [num_comment, comment_len], each element is int32)
    # Terms are already converted into indexes
    return load_pickled_data("code_comments.pickle")

def load_articles():
    # List[Short_id, documment]
    return load_pickled_data("code_articles.pickle")

def load_guardian_labeled():
    data = load_pickled_data("eval_data.pickle")
    x_doc, x_comments, y = zip(*data)
    return list(x_doc), list(x_comments), list(y)

def load_voca():
    return load_pickled_data('word2idx')

def save_reaction_label_cache(label_dict, id):
    path = os.path.join(data_path, "r_{}.pickle".format(id))
    pickle.dump(label_dict, open(path, "wb"))

def load_stopwords():
    s = set()
    f = open(os.path.join(data_path, "stopwords.dat"), "r")
    for line in f:
        s.add(line.strip())
    return s

def load_init_embbedding():
    return load_pickled_data("init_embedding.pickle")

def save_pickle_data(obj, name):
    path = os.path.join(data_path, name)
    return pickle.dump(obj, open(path, "wb"))

def tokenize(sentence):
    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r" \'(.*)\'([ \.])", r" \1\2", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    tokens = clean_str(sentence).split(" ")
    return tokens


def tokenize_encode(text, voca2idx, comment_length):
    OOV = 1
    PADDING = 0
    result = []
    for token in tokenize(text)[:comment_length]:
        idx = voca2idx[token] if token in voca2idx else OOV
        result.append(idx)

    result = result + [PADDING] * (comment_length-len(result))
    return result



def format_comment(comments, voca2idx, comment_length):
    # comments : List[(Content, Label)]
    X = []
    Y = []
    for text, label in comments:
        X.append(tokenize_encode(text, voca2idx, comment_length))
        Y.append(label)
    return np.array(X), np.array(Y)


agree_path = os.path.join(data_path,"commentAgree.txt")

# 0 : neutral
# 1 : agree
# 2 : disagree
def load_agree_data():
    comments = []
    for line in open(agree_path, encoding="utf-8"):
        label = int(line[:1])
        content = line[2:]
        yield (content, label)

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