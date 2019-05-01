from collections import Counter
from load_data import *
from multiprocessing import Pool, freeze_support
from text_rank import TextRank

def reverse_voca(word2idx):
    OOV = 1
    PADDING = 0
    idx2word = dict()
    idx2word[1] = "OOV"
    idx2word[0] = "PADDING"
    idx2word[3] = "LEX"
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return idx2word

def arr2str(arr):
    return " ".join([str(item) for item in arr])

def str2arr(s):
    return [int(item) for item in s.split(" ")]

class PhraseBox:
    def __init__(self):
        self.idx2word = reverse_voca(load_voca())
        self.phrases = load_pickled_data('g_phrase.pickle')
        # phrases : List["123 52 181"]
        self.set_phrase = set(self.phrases)

    def to_text(self, phrase):
        indices = [int(item) for item in phrase.split(" ")]
        if not all([idx in self.idx2word for idx in indices]):
            return None
        return " ".join([self.idx2word[idx] for idx in indices])

    def contains(self, list_index):
        return arr2str(list_index) in self.set_phrase



class CorpusInfo:
    def __init__(self):
        self.OOV = 1
        self.PADDING = 0
        self.voca = load_voca()
        self.good_phrase = PhraseBox()



def top_phrase_by_scorer(params):
    doc, scorer, corpus_info, text_rank, k = params
    tr_score = text_rank.run(doc)

    cont_score = Counter()
    for i in range(len(doc)):
        st = i
        for phrase_l in range(1, 4):
            ed = i + phrase_l
            n_gram = doc[st:ed]
            if corpus_info.good_phrase.contains(n_gram):
                phrase = arr2str(n_gram)
                score = sum([scorer(token) for token in n_gram])
                cont_score[phrase] += score

    per_doc_score = Counter()
    for phrase, score in cont_score.most_common(40):
        factor = sum([tr_score[token] for token in str2arr(phrase) if token in tr_score])
        per_doc_score[phrase] += factor * score

    return list(per_doc_score.most_common(4))

def get_textrizer_plain(word2idx):
    idx2word = reverse_voca(word2idx)
    def textrize(indice):
        text = []
        PADDING = 0
        for i in range(len(indice)):
            word = idx2word[indice[i]]
            if indice[i] == PADDING:
                break
            text.append(word)
        return " ".join(text)
    return textrize

def top_topics_from_lm():
    print("top_topics_from_lm")
    LM = load_pickled_data("LM_classifier_10.pickle")
    voca = load_voca()
    corpus_info = CorpusInfo()
    articles = load_articles()
    ids, docs = zip(*articles)

    print("Predicting")
    labels = LM.predict_parallel(docs)
    cont_ids, _ = zip(*filter(lambda x:x[1], zip(ids, labels)))
    cont_ids = set(cont_ids)
    print("{}% are controversial".format(len(cont_ids)/len(docs)))
    save_pickle_data(cont_ids, "cont_ids")

    print("Initialize TextRank")
    text_rank = TextRank(docs, voca)
    scorer = LM.token_odd

    payloads = []
    for id, doc in articles:
        if id in cont_ids:
            param = (doc, scorer, corpus_info, text_rank, 4)
            payloads.append(param)

    n_thread = 30
    p = Pool(n_thread)
    g_phrase_score = Counter()
    print("Mapping")
    for phrase_score in p.map(top_phrase_by_scorer, payloads):
        for phrase, score in phrase_score:
            g_phrase_score[phrase] += score

    textrize = get_textrizer_plain(voca)

    result = []
    for phrase, score in g_phrase_score.most_common(300):
        plain_phrase = textrize(str2arr(phrase))
        print("{}\t{}".format(plain_phrase, score))
        result.append((plain_phrase,score))
    save_pickle_data(result, "cont_topics_lm.pickle")


if __name__ == "__main__":
    top_topics_from_lm()
