import math
from collections import Counter
from multiprocessing import Pool

import numpy as np


def count_word(documents):
    PADDING = 0
    counter = Counter()
    for doc in documents:
        for i, token in enumerate(doc):
            if token == PADDING:
                break
            counter[token] += 1
    return counter


class LM:
    per_odd = dict()

    def __init__(self, per_odd, alpha):
        self.per_odd = per_odd
        self.alpha = alpha

    @classmethod
    def train(cls, c_docs, nc_docs):
        c_counter = cls.count_word_parallel(c_docs)
        nc_counter = cls.count_word_parallel(nc_docs)
        per_odd = cls.evaluate_per_token_odd(c_counter, nc_counter)
        alpha = cls.get_alpha(per_odd, c_docs, nc_docs)
        return cls(per_odd, alpha)

    @classmethod
    def load(cls, per_odd):
        return cls(per_odd)


    @classmethod
    def get_alpha(cls, per_odd, c_docs, nc_docs):
        vectors = []
        for doc in c_docs:
            odd = cls.doc_odd(per_odd, doc)
            vectors.append((odd, 1))
        for doc in nc_docs:
            odd = cls.doc_odd(per_odd, doc)
            vectors.append((odd, 0))

        vectors.sort(key=lambda x:x[0], reverse=True)

        total = len(vectors)
        p =  len(c_docs)
        fp = 0

        max_acc = 0
        opt_alpha = 0
        for idx, (odd, label) in enumerate(vectors):
            alpha = odd - 1e-8
            if label == 0:
                fp += 1

            tp = (idx+1) - fp
            fn = p - tp
            tn = total - (idx+1) - fn
            acc = (tp + tn) / (total)
            if acc > max_acc:
                opt_alpha = alpha
                max_acc = acc
        print("alpha(Threshold) : {}".format(opt_alpha))
        print("Train acc : {}".format(max_acc))
        return opt_alpha



    @classmethod
    def chunks(cls, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @classmethod
    def count_word_parallel(cls, documents):
        split = 30
        p = Pool(split)
        args = cls.chunks(documents, split)
        counters = p.map(count_word, args)
        g_counter = Counter()
        for counter in counters:
            for key in counter.keys():
                g_counter[key] += counter[key]
        return g_counter

    @staticmethod
    def evaluate_per_token_odd(c_counter, nc_counter):
        def count(LM, token):
            if token in LM:
                return LM[token]
            else:
                return 0

        c_ctf = sum(c_counter.values())
        nc_ctf = sum(nc_counter.values())
        smoothing = 0.1
        def per_token_odd(token):
            tf_c = count(c_counter, token)
            tf_nc = count(nc_counter, token)
            if tf_c == 0 and tf_nc == 0 :
                return 0
            P_w_C = tf_c / c_ctf
            P_w_NC = tf_nc / nc_ctf
            P_w_BG = (tf_c+tf_nc) / (nc_ctf + c_ctf)
            logC = math.log(P_w_C * smoothing + P_w_BG * (1-smoothing))
            logNC = math.log(P_w_NC * smoothing + P_w_BG * (1 - smoothing))
            assert(math.isnan(logC)==False)
            assert(math.isnan(logNC)== False)
            return logC - logNC

        per_odd = dict()
        all_word = set(c_counter.keys()) | set(nc_counter.keys())
        for token in all_word:
            per_odd[token] = per_token_odd(token)
        return per_odd

    @classmethod
    def doc_odd(cls, per_odd, document):
        odd_sum = 0
        for token in document:
            if token in per_odd:
                odd_sum += per_odd[token]
        return odd_sum

    def token_odd(self, token):
        if token in self.per_odd:
            return self.per_odd[token]
        else :
            return 0

    def cont_score(self, document):
        return self.doc_odd(self.per_odd, document)

    def predict(self, document):
        return self.cont_score(document) > self.alpha

    def predict_parallel(self, documents):
        p = Pool(30)
        y_list = p.map(self.predict, documents)
        return y_list


    def predict_documents(self, documents):
        return np.array([(self.alpha, self.doc_odd(self.per_odd, d)) for d in documents])