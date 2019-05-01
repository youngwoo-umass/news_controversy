from optimizer import *
from load_data import *


def em_full_iteration():
    print("lm_full_iteration")
    articles = load_articles()
    reaction_list = load_reactions()
    eval_data = load_guardian_labeled()
    print("Step 1) Train AD classifier")
    comment_label_dict = predict_by_disagreement(reaction_list, True)
    print("Step 2) Train doc LM")
    doc_LM = train_article_classifier1(articles, comment_label_dict, eval_data)
    itr = 3
    while itr < 10:
        print("Step {}) Train comment LM".format(itr))
        comment_LM = train_comment_LM(reaction_list, articles, doc_LM, eval_data)
        itr += 1
        print("Step {}) Train doc LM".format(itr))
        doc_LM = train_doc_by_LM(reaction_list, articles, comment_LM, eval_data)
        save_pickle_data(doc_LM, "LM_classifier_{}.pickle".format(itr))
        itr += 1


if __name__ == "__main__":
    em_full_iteration()