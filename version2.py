#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import os
import re
import logging
import pandas as pd
import numpy as np
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from sklearn import naive_bayes, svm, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
# import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import time


# Pre-processing movie reviews
def clean_review(raw_review):
    # Remove HTML markup
    text = BeautifulSoup(raw_review, features="html.parser")

    # Removing digits and punctuation
    text = re.sub("[^a-zA-Z]", " ", text.get_text())

    # Converting to lowercase
    text = text.lower().split()

    # Removing stopwords
    stops = set(stopwords.words("english"))
    words = [w for w in text if w not in stops]

    # Return a cleaned string
    return " ".join(words)


# Generates a feature vector(word2vec averaging) for each movie review
def review_to_vec(words, model, num_features):
    """
    This function generates a feature vector for the given review.
    Input:
        words: a list of words extracted from a review
        model: trained word2vec model
        num_features: dimension of word2vec vectors
    Output:
        a numpy array representing the review
    """

    feature_vec = np.zeros((num_features), dtype="float32")
    word_count = 0

    # index2word_set is a set consisting of all words in the vocabulary
    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            word_count += 1
            feature_vec += model[word]

    feature_vec /= word_count
    return feature_vec


# Generates vectorized movie reviews
def gen_review_vecs(reviews, model, num_features):
    """
    Function which generates a m-by-n numpy array from all reviews,
    where m is len(reviews), and n is num_feature
    Input:
            reviews: a list of lists.
                     Inner lists are words from each review.
                     Outer lists consist of all reviews
            model: trained word2vec model
            num_feature: dimension of word2vec vectors
    Output: m-by-n numpy array, where m is len(review) and n is num_feature
    """

    curr_index = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

        if curr_index % 1000 == 0.:
            print("Vectorizing review %d of %d" % (curr_index, len(reviews)))

        review_feature_vecs[curr_index] = review_to_vec(review, model, num_features)
        curr_index += 1

    return review_feature_vecs


# TFIDF vectorization
def tfidf_vectorizer(train_list, test_list, train_data, test_data):
    stop_words_list = ["ke","ka","ek","mein","kee", "hai","yah","aur","se","hain","ko","par","is","hota","ki", "jo","kar", "me", "gaya", "karane", "kiya","liye",
                      "apane", "ne", "banee", "nahin", "to", "hee","ya","evan", "diya", "ho","isaka", "tha","dvaara", "hua","tak","saath","karana","vaale", "baad",
                      "lie", "aap", "kuchh", "sakate","kisee", "ye", "isake", "sabase", "isamen", "the", "do", "hone","vah","ve","karate","bahut","kaha", "varg",
                      "kaee","karen","hotee", "apanee","unake", "thee", "yadi","huee","ja", "na", "ise", "kahate", "jab","hote", "koee","hue", "va", "na","abhee",
                      "jaise","sabhee", "karata", "unakee", "tarah", "us", "aadi", "kul", "es","raha", "isakee","sakata", "rahe","unaka","isee","rakhen", "apana", "pe","usake"]
    for i in range(0, len(train_data.review)):

        # Append raw texts as TFIDF vectorizers take raw texts as inputs
        train_list.append(clean_review(train_data.review[i]))
        if i % 1000 == 0:
            print("Cleaning training review", i)

    for i in range(0, len(test_data.review)):

        # Append raw texts as TFIDF vectorizers take raw texts as inputs
        test_list.append(clean_review(test_data.review[i]))
        if i % 1000 == 0:
            print("Cleaning test review", i)
        # print(clean_review(test_data.review[i]))
    count_vec = TfidfVectorizer(min_df=0,max_df=0.6,ngram_range=(0, 1), sublinear_tf=True,norm='l2',stop_words=None)
    print("Vectorizing input texts")
    train_vec = count_vec.fit_transform(train_list)
    test_vec = count_vec.transform(test_list)
    print(test_vec.shape)
    return train_vec, test_vec, count_vec


# Performing dimensionality reduction using SelectKBest
def dimensionality_reduction(train_vec, test_vec, y_train_data):
    print("Performing feature selection based on chi2 independence test")
    fselect = SelectKBest(chi2, k=4500)
    train_vec = fselect.fit_transform(train_vec, y_train_data)
    test_vec = fselect.transform(test_vec)
    return train_vec, test_vec

from sklearn import metrics
# Multinomial Naive Bayes classifier
def naive_bayes(train_vec, test_vec, y_train_data):
    start = time.time()
    nb = MultinomialNB(alpha=1)
    # param = {'alpha':[1e-1,0.3,0.7,0.9,1.1]}
    # nb = GridSearchCV(nb, param, cv=5, scoring="roc_auc")
    # nb.fit(train_vec,y_train_data)
    # print("best_score:",nb.best_score_)
    # print("best_etimator: ", nb.best_params_)
    cv_score = cross_val_score(nb, train_vec, y_train_data, cv=5,scoring="roc_auc")
    print("Training Multinomial Naive Bayes")
    nb = nb.fit(train_vec, y_train_data)
    pred_naive_bayes = nb.predict(test_vec)

    # print(pred_naive_bayes)
    print("CV Score = ", cv_score)
    print("Total time taken for Multinomial Naive Bayes is ", time.time() - start, " seconds")
    result = nb.predict_proba(test_vec)
    result = [i[1] for i in result]
    test_auc = metrics.roc_auc_score(np.array(test_label1), result)  # 验证集上的auc值
    print("test_auc:",test_auc)

    # print(result[:10])
    return pred_naive_bayes, result


# Random Forest classifier
def random_forest(train_vec, test_vec, y_train_data):
    start = time.time()
    rfc = RFC(n_estimators=200, oob_score=True, max_features="auto")
    cv_score = cross_val_score(rfc, train_vec, y_train_data, cv=5, scoring="roc_auc")
    print("Training %s" % ("Random Forest"))
    rfc = rfc.fit(train_vec, y_train_data)

    print("OOB Score =",cv_score)
    pred_random_forest = rfc.predict(test_vec)
    print("Total time taken for Random Forest is ", time.time() - start, " seconds")

    return pred_random_forest


# Linear SVC classifier
def linear_svc(train_vec, test_vec, y_train_data):
    start = time.time()
    svc = svm.LinearSVC(C=0.8, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='ovr', max_iter=1000,
     multi_class='crammer_singer', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
    # svc = CalibratedClassifierCV(svc)

    # param = {'max_iter': [10000], 'C': [1e15, 1e13, 1e11, 1e9, 1e7, 1e5, 1e3, 1e1, 1e-1, 1e-3, 1e-5]}
    # param = {'max_iter': [600,800,1000], 'C': [ 1e5,1e3, 1e1, 1e-1, 1e-3]}
    # print("Training SVC")
    # svc = GridSearchCV(svc, param, cv=5)
    # svc = svc.fit(train_vec, y_train_data)
    # # pred_linear_svc = svc.predict(test_vec)
    # print("Optimized parameters:", svc.best_estimator_)
    # print("Best CV score:", svc.best_score_)
    # print("Total time taken for Linear SVC is ", time.time() - start, " seconds")
    # print("Generating confusion matrix")
    svc1 = CalibratedClassifierCV(svc)
    svc1.fit(train_vec,y_train_data)
    result = svc1.predict_proba(test_vec)

    svc_score = cross_val_score(svc,train_vec,y_train_data,cv=5,scoring="roc_auc")
    print("svc roc:",svc_score)
    svc = svc.fit(train_vec, y_train_data)
    pred_linear_svc = svc.predict(test_vec)

    result = [i[1] for i in result]
    test_auc = metrics.roc_auc_score(np.array(test_label1), result)  # 验证集上的auc值
    print("test_auc:", test_auc)
    # Below confusion matrix code is commented as it takes a lot of time to run. The plots have been added in the project report.
    # predictions = cross_val_predict(svc, train_vec, y_train_data)
    # skplt.metrics.plot_confusion_matrix(y_train_data, predictions)
    # plt.show()
    print(result[:10])
    return pred_linear_svc,result


# SVM Classifier using cross validation
def svm_cross_validation(train_x, test_ver, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [0.01,0.1,1, 10, 100, 1000,1500,2000], 'gamma': [0.1,0.01,0.001]}
    grid_search = GridSearchCV(model, param_grid, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    print("svc best score:",grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    predict_res = model.predict(test_ver)
    print(predict_res[:10])

    return predict_res


# Logistic Regression
def logistic_regression(train_vec, test_vec, y_train_data):
    start = time.time()
    # clf = LogisticRegression(random_state=0, solver='liblinear', max_iter=10000, multi_class='multinomial')
    clf = LogisticRegression(random_state=0, solver='newton-cg', max_iter=5000, multi_class='multinomial',class_weight='balanced',C=1.5)
    cv_score = cross_val_score(clf, train_vec, y_train_data, cv=5, scoring='roc_auc')
    print("Training Logistic Regression")
    clf = clf.fit(train_vec, y_train_data)
    pred_logistic = clf.predict(test_vec)
    print("CV Score = ", cv_score)
    print("Total time taken for Logistic is ", time.time() - start, " seconds")
    print("Plotting Precision recall curve")
    result = clf.predict_proba(test_vec)
    # print(result[:15])
    # skplt.metrics.plot_precision_recall(y_train_data, result)
    result = [i[1] for i in result]
    test_auc = metrics.roc_auc_score(np.array(test_label1), result)  # 验证集上的auc值
    print("test_auc:", test_auc)
    # plt.show()
    return pred_logistic, result

import xgboost as xgb
def xgboost_test(train_vec, test_vec, y_train_data):
    dtrain = xgb.DMatrix(train_vec, label=y_train_data)
    dtest = xgb.DMatrix(test_vec)
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1}
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=5, evals=watchlist)
    # # 输出概率
    # ypred = bst.predict(dtest)
    # print(ypred)
    # print(type(train_vec))
    # print(type(y_train_data))
    # data_train = xgb.DMatrix(train_vec, label=y_train_data)
    #
    # data_test = xgb.DMatrix(test_vec, label=test_label1)
    # watch_list = [(data_test, 'eval'), (data_train, 'train')]
    # params = {'max_depth': 1, 'eta': 0.9, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    # bst = xgb.train(params, data_train, num_boost_round=7, evals=watch_list)
    # y_train_pred = bst.predict(data_train)
    # y_test_pred = bst.predict(data_test)
    # print('XGBoost训练集准确率：', accuracy_score(y_train_data, y_train_pred))
    # print('XGBoost测试集准确率：', accuracy_score(test_label1, y_test_pred))
    # return y_test_pred



# Word2Vec vectorization
def word2vec(train_data, test_data, train_list, test_list):
    model_name = "GoogleNews-vectors-negative300.bin.gz"
    model_type = "bin"
    num_features = 300
    for i in range(0, len(train_data.review)):
        train_list.append(clean_review(train_data.review[i]))
        if i % 1000 == 0:
            print("Cleaning training review", i)
    for i in range(0, len(test_data.review)):
        test_list.append(clean_review(test_data.review[i]))
        if i % 1000 == 0:
            print("Cleaning test review", i)
    print("Loading the pre-trained model")
    # The below part has been commented as the model was loaded, movie reviews were vectorized and stored in below pkl files,
    # as this takes a lot of time to execute.
    # We are reading the pkl files to get the final vectorized data

    # model = Word2Vec.load_word2vec_format(model_name, binary=True)
    print("Vectorizing training review")
    # train_vec = gen_review_vecs(train_list, model, num_features)
    # print ("Vectorizing test review")
    # test_vec = gen_review_vecs(test_list, model, num_features)

    # print("Writing to DataFrame after vectorizing")
    # df_train = pd.DataFrame(train_vec)
    # df_test = pd.DataFrame(test_vec)
    # df_train.to_pickle("train.pkl")
    # df_test.to_pickle("test.pkl")


    y_train_data = train_data.sentiment
    train_df = pd.read_pickle("train.pkl")
    test_df = pd.read_pickle("test.pkl")

    # Word2Vec cannot be used with Multinomial Naive Bayes as Multinomial Naive Bayes does not work with negative values
    pred_logistic = logistic_regression(train_df, test_df, y_train_data)
    pred_random_forest = random_forest(train_df, test_df, y_train_data)
    pred_linear_svc ,result= linear_svc(train_df, test_df, y_train_data)

    output = pd.DataFrame(data={"id": test_data.id, "Pre": [result[0] for i in result]})
    output.to_csv("word2vec_svc.csv", index=False)


# Testing a custom movie review
def test_custom_review(count_vec, train_vec, y_train_data):
    print('\nTest a custom review message')
    print('Enter review to be analysed: ', end=" ")

    test = []
    test_list = []
    test.append(input())
    test_review = pd.DataFrame(data={"id": 1, "review": test})
    print("Cleaning the test review")
    for i in range(0, len(test_review.review)):
        test_list.append(clean_review(test_review.review[i]))
    print("Vectorizing the test review")
    test_review_vec = count_vec.transform(test_list)
    print("Predicting")
    pred_naive_bayes = naive_bayes(train_vec, test_review_vec, y_train_data)
    if (pred_naive_bayes == 1):
        print("The review is predicted positive")
    else:
        print("The review is predicted negative")


if __name__ == "__main__":
    from sklearn.preprocessing import OneHotEncoder

    from sklearn.preprocessing import LabelEncoder
    train_list = []
    test_list = []
    word2vec_input = []

    pred_naive_bayes = []
    pred_logistic = []
    pred_random_forest = []
    pred_linear_svc = []
    # train_data = pd.read_csv("train.csv", header=0, delimiter="\t", quoting=0)
    train_data = pd.read_csv("./train.csv", lineterminator='\n')
    # test_data = pd.read_csv("20190506_test.csv", header=0, delimiter="\t", quoting=0)
    test_data = pd.read_csv("./20190513_test.csv", lineterminator='\n')
    # print(test_data[:10])
    y_train_data = train_data.label


    # onehot_encoder = OneHotEncoder(sparse=False)
    # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    # print(y_train_data[:10])
    # print(train_data.shape[0])
    train_data1 = train_data[int(0.2*train_data.shape[0]):]
    test_data1 = train_data[:int(0.2*train_data.shape[0])]
    # print(test_data1.shape[0])
    train_data1 = train_data1.reset_index(drop=True)
    # test_data1 = pd.DataFrame(test_data1,index=[1,2])
    # for i in range(test_data1.shape[0]):
    #     test_data1.iloc[i,0] = i+1
    # print(test_data1)
    train_label1 = y_train_data[int(0.2*train_data.shape[0]):]
    test_label1 = y_train_data[:int(0.2*train_data.shape[0])]
    train_label1=train_label1.reset_index(drop=True)
    # print(test_label1[:10])
    label_encoder = LabelEncoder()
    train_label1 = label_encoder.fit_transform(train_label1)
    print(train_label1.shape)

    label_encoder = LabelEncoder()
    test_label1 = label_encoder.fit_transform(test_label1)
    print(test_label1.shape)

    # Vectorization - TFIDF
    print("Using TFIDF ")
    train_vect, test_vec, count_vec = tfidf_vectorizer(train_list, test_list, train_data1, test_data1)

    print(train_vect.shape)

    # Dimensionality Reduction
    # train_vect, test_vec = dimensionality_reduction(train_vect, test_vec, train_label1)
    # cc = xgboost_test(train_vect, test_vec, train_label1)
    # train_vec1 = train_vec[:9*len(train_vec)]
    # y_train_data1 = y_train_data[:9*len(y_train_data)]
    # test_vec1 = train_vec[9*len(train_vec):]
    # y_test_data1 = y_train_data[9*len(y_train_data):]

    # Prediction
    # pred_naive_bayes, result1 = naive_bayes(train_vect, test_vec, train_label1)
    # pred_random_forest = random_forest(train_vect, test_vec, train_label1)

    # pred_logistic, result2 = logistic_regression(train_vect, test_vec, train_label1)

    # pred_linear_svc, result3= linear_svc(train_vect, test_vec, train_label1)

    # result4 = (np.array(result1)+np.array(result2)+np.array(result3))/3
    # test_auc = metrics.roc_auc_score(np.array(test_label1), result4)  # 验证集上的auc值
    # print("test_auc_end:", test_auc)
    # pre_svc = svm_cross_validation(train_vec,test_vec,y_train_data)
    # Writing output of classifier with highest accuracy(Linear SVC)to csv
    # print(result4[:10])
    # output = pd.DataFrame(data={"ID": test_data.ID, "Pred": [i for i in result4]})

    # output.to_csv("tfidf_svc.csv", index=False)

    # print("Using pre-trained word2vec model")
    # train_list = []
    # test_list = []
    # pred_logistic = []
    # pred_random_forest = []
    # pred_linear_svc = []
    #
    # word2vec(train_data, test_data, train_list, test_list)
    #
    # # Test a custom review using Multinomial Naive Bayes
    # test_custom_review(count_vec, train_vect, y_train_data)