""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.99, k1=1.6, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(max_df=0.90, min_df=1,
                                          use_idf=True,
                                          ngram_range=ngram_range,)

        # print(ngram_range)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])

        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)

        return (numer / denom).sum(1).A1

# # Evaluation
# Precision@K Function
def prec_at_k(true_list,pred_list,k):
    # define list of top k predictions
    count = 0
    top_k_pred = pred_list[0:k]
    # iterate throught the top k predictions
    for doc in top_k_pred:
        # if document in true list, then increment count of relevant predictions
        if doc in true_list:
            count += 1
    # return total_relevant_predictions_in_top_k/k
    return count/k


# Recall@K Function
def recall_at_k(true_list,pred_list,k,r):
    # define top k predictions
    count = 0
    top_k_pred = pred_list[0:k]
    # iterate through the top k predictions
    for doc in top_k_pred:
        # if doc in true list, then increment count
        if doc in true_list:
            count += 1
    # return number of relevant documents in top k predictions/total number of relevant predictions
    return count/r


# Average Precision Function
def AP(true_list,pred_list):
    # P-> relative precision list, rel_vec-> relevance vector
    P = []
    rel_vec = []
    val = 0
    # iterate through the entire prediction list
    for i in range(len(pred_list)):
        # if predicted citation in true list increment numberator (number of relevant docs) by 1 and also append 1 for rel_vec
        if pred_list[i] in true_list:
            val += 1
            rel_vec.append(1)
        else:
            # otherwise just append 0 for rel_vec
            rel_vec.append(0)
        # append the relative precision for each query document while iterating
        # so append (number of relevant docs so far ie., val) divided by total number of documents iterated so far
        P.append(val/(i+1))
    count = 0
    total = 0
    # find the relatve precision of all the relevant documents and take sum
    for rank in range(len(P)):
        # for index in P list
        # if rel_vec[i] is 1 that means it is relevant document thus increment count and add to total, else dont count
        if rel_vec[rank] == 1:
            count += 1
            total += P[rank]
    # boundary case where there is no relevent document found
    if count == 0:
        return 0
    # return the Average Precision
    return total/count


# Reciprocal Rank Function
def RR(true_list,pred_list):
    # iterate through the ranked prediction list, break at first relevant case and return reciprocal of that rank
    for i in range(len(pred_list)):
        if pred_list[i] in true_list:
            return 1/(i+1)


#Micro Precision Function
def micro_prec(true_list,pred_list,k):
    #define list of top k predictions
    cor_pred = 0
    top_k_pred = pred_list[0:k].copy()
    #iterate throught the top k predictions
    for doc in top_k_pred:
        #if document in true list, then increment count of relevant predictions
        if doc in true_list:
            cor_pred += 1
    #return total_relevant_predictions_in_top_k/k
    return cor_pred, k  