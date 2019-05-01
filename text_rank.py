import collections
from load_data import load_stopwords


class TextRank:
    def __init__(self, data, voca):
        self.p_reset = 0.1
        self.p_transition = 1 - self.p_reset
        self.max_repeat = 500
        self.window_size = 10
        self.idf = collections.Counter()
        self.def_idf = 2
        for document in data:
            for word in set(document):
                self.idf[word] += 1
        self.stopword = set()
        for word in load_stopwords():
            if word in voca:
                self.stopword.add(voca[word])


    def get_edge(self, vertice, token_doc):
        raw_count = dict((vertex, collections.Counter()) for vertex in vertice)
        for i in range(len(token_doc)):
            source = token_doc[i]
            st = max(i - int(self.window_size / 2), 0)
            ed = min(i + int(self.window_size / 2), len(token_doc))
            for j in range(st, ed):
                target = token_doc[j]
                raw_count[source][target] += 1

        edges = dict()
        for vertex in vertice:
            out_sum = sum(raw_count[vertex].values())
            out_weights = dict()
            for target in raw_count.keys():
                out_weights[target] = raw_count[vertex][target] / out_sum
            edges[vertex] = out_weights
        return edges

    def get_reset(self, token_doc):
        tf_d = collections.Counter(token_doc)
        d = dict()
        for word, tf in tf_d.items():
            if word in self.idf:
                d[word] = tf / self.idf[word]
            else:
                d[word] = tf / self.def_idf
        total = sum(d.values())
        for word in tf_d.keys():
            d[word] = d[word] / total
        return d

    def run(self, raw_token_doc):
        def not_stopword(word):
            return not word in self.stopword
        token_doc = list(filter(not_stopword, raw_token_doc))
        vertice = set(token_doc)
        v_reset = self.get_reset(token_doc)
        edges = self.get_edge(vertice, token_doc)
        def same(before, after):
            same_thres = 1e-3
            assert(len(before) == len(after))
            before.sort(key=lambda x:x[0])
            after.sort(key=lambda x:x[0])
            for b, a in zip(before, after):
                (word_b, p_b) = b
                (word_a, p_a) = a
                assert(word_a == word_b)
                if abs(p_b-p_a) > same_thres:
                    return False
            return True

        init_p = 1 / len(vertice)
        p_vertice = [(vertex, init_p) for vertex in vertice]

        n_repeat = 0
        while n_repeat < self.max_repeat:
            p_vertice_next = dict((vertex, self.p_reset * init_p) for vertex in vertice)
            for vertex, p in p_vertice:
                for target, edge_p in edges[vertex].items():
                    p_vertice_next[target] += edge_p * p * self.p_transition
            p_vertice_next = list(p_vertice_next.items())
            n_repeat += 1
            if same(p_vertice, p_vertice_next):
                break
            p_vertice = p_vertice_next

        return dict((vertex, p) for vertex, p in p_vertice)
