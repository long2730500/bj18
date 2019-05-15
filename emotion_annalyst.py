# import numpy as np
# import tensorflow as tf
# wordsList = np.load('./training_data/wordsList.npy')
# print('Loaded the word list!')
# wordsList = wordsList.tolist() #Originally loaded as numpy array
# wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
# wordVectors = np.load('./training_data/wordVectors.npy')
#
# from os import listdir
# from os.path import isfile, join
# positiveFiles = ['./training_data/positiveReviews/' + f for f in listdir('./training_data/positiveReviews/') if isfile(join('./training_data/positiveReviews/', f))]
# negativeFiles = ['./training_data/negativeReviews/' + f for f in listdir('./training_data/negativeReviews/') if isfile(join('./training_data/negativeReviews/', f))]
# numWords = []
# for pf in positiveFiles:
#     with open(pf, "r", encoding='utf-8') as f:
#         line=f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Positive files finished')
#
# for nf in negativeFiles:
#     with open(nf, "r", encoding='utf-8') as f:
#         line=f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Negative files finished')
#
# numFiles = len(numWords)
# print('The total number of files is', numFiles)
# print('The total number of words in the files is', sum(numWords))
# print('The average number of words in the files is', sum(numWords)/len(numWords))
#
# # 删除标点符号、括号、问号等，只留下字母数字字符
# import re
# strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
#
# def cleanSentences(string):
#     string = string.lower().replace("<br />", " ")
#     return re.sub(strip_special_chars, "", string.lower())
#
# fname = positiveFiles[3] #Can use any valid index (not just 3)
# maxSeqLength = 250
# numDimensions = 300
# firstFile = np.zeros((maxSeqLength), dtype='int32')
# with open(fname) as f:
#     indexCounter = 0
#     line=f.readline()
#     cleanedLine = cleanSentences(line)
#     split = cleanedLine.split()
#     for word in split:
#         try:
#             firstFile[indexCounter] = wordsList.index(word)
#         except ValueError:
#             firstFile[indexCounter] = 399999 #Vector for unknown words
#         indexCounter = indexCounter + 1
#
# # ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# # fileCounter = 0
# # for pf in positiveFiles:
# #    with open(pf, "r") as f:
# #        indexCounter = 0
# #        line=f.readline()
# #        cleanedLine = cleanSentences(line)
# #        split = cleanedLine.split()
# #        for word in split:
# #            try:
# #                ids[fileCounter][indexCounter] = wordsList.index(word)
# #            except ValueError:
# #                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
# #            indexCounter = indexCounter + 1
# #            if indexCounter >= maxSeqLength:
# #                break
# #        fileCounter = fileCounter + 1
#
# # for nf in negativeFiles:
# #    with open(nf, "r") as f:
# #        indexCounter = 0
# #        line=f.readline()
# #        cleanedLine = cleanSentences(line)
# #        split = cleanedLine.split()
# #        for word in split:
# #            try:
# #                ids[fileCounter][indexCounter] = wordsList.index(word)
# #            except ValueError:
# #                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
# #            indexCounter = indexCounter + 1
# #            if indexCounter >= maxSeqLength:
# #                break
# #        fileCounter = fileCounter + 1
# # #Pass into embedding function and see if it evaluates.
#
# # np.save('idsMatrix', ids)
#
# ids = np.load('./training_data/idsMatrix.npy')
#
# from random import randint
#
# def getTrainBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         if (i % 2 == 0):
#             num = randint(1,11499)
#             labels.append([1,0])
#         else:
#             num = randint(13499,24999)
#             labels.append([0,1])
#         arr[i] = ids[num-1:num]
#     return arr, labels
#
# def getTestBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = randint(11499,13499)
#         if (num <= 12499):
#             labels.append([1,0])
#         else:
#             labels.append([0,1])
#         arr[i] = ids[num-1:num]
#     return arr, labels
#
# # RNN模型
# batchSize = 24
# lstmUnits = 64
# numClasses = 2
# iterations = 50000
#
# tf.reset_default_graph()
#
# labels = tf.placeholder(tf.float32, [batchSize, numClasses])
# input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
#
# data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
# data = tf.nn.embedding_lookup(wordVectors,input_data)
#
# lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
# value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
#
# weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
# bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
# value = tf.transpose(value, [1, 0, 2])
# #取最终的结果值
# last = tf.gather(value, int(value.get_shape()[0]) - 1)
# prediction = (tf.matmul(last, weight) + bias)
#
# correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
# accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
#
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
# optimizer = tf.train.AdamOptimizer().minimize(loss)
#
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
#
# for i in range(iterations):
#     # Next Batch of reviews
#     nextBatch, nextBatchLabels = getTrainBatch()
#     sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#
#     if (i % 1000 == 0 and i != 0):
#         loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
#         accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
#
#         print("iteration {}/{}...".format(i + 1, iterations),
#               "loss {}...".format(loss_),
#               "accuracy {}...".format(accuracy_))
#         # Save the network every 10,000 training iterations
#     if (i % 10000 == 0 and i != 0):
#         save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
#         print("saved to %s" % save_path)



import nltk
import pandas as pd
import numpy as np

train_raw_data = pd.read_csv("./train.csv",lineterminator='\n')
test_raw_data = pd.read_csv("./20190506_test.csv",lineterminator='\n')
print(train_raw_data.head(10))
print(type(train_raw_data.iloc[1, 1]))

def format_sentence(sent):
    return({word:True for word in nltk.word_tokenize(sent)})
test_data = []
train_data = []
for i in range(train_raw_data.shape[0]):
    train_data.append([format_sentence(train_raw_data.iloc[i, 1]), train_raw_data.iloc[i, 2]])

# print(train_data[:10])

training = train_data[:int((.85)*len(train_data))]

test = train_data[int((.85)*len(train_data)):]

from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(training)
# classifier.show_most_informative_features()




# print(test[1][0])

# predict

predict_test=[]
print(test_raw_data.shape)
res = pd.DataFrame(np.zeros([test_raw_data.shape[0],2],dtype=int),columns=list(["ID","Pred"]))
for i in range(test_raw_data.shape[0]):
    predict_test.append(format_sentence(test_raw_data.iloc[i, 1]))

for i in range(test_raw_data.shape[0]):
    probdist = classifier.prob_classify(predict_test[i])
    pre_sentiment = probdist.max()
    # print(pre_sentiment)
    if pre_sentiment == "Positive":
        aa=round(probdist.prob(pre_sentiment),2)
    else:
        if probdist.prob(pre_sentiment)<0.7:
            aa=round(probdist.prob(pre_sentiment),2)
        else:
            aa = round(1-probdist.prob(pre_sentiment),2)
    res.iloc[i, 0] = int(i)+1
    res.iloc[i, 1] = aa
# res.reindex = ["ID"]
res.to_csv("Result.csv",index=0)
print(res.head(10))
from nltk.classify.util import accuracy
print(accuracy(classifier, test))

import sklearn.naive_bayes

# def format_sentence(sent):
#     return({word:True for word in nltk.word_tokenize(sent)})
#
# print(format_sentence("The cat is very cute"))
#
# pos = []
# with open("./pos_tweets.txt",'r', encoding='UTF-8') as f:
#     for i in f:
#         pos.append([format_sentence(i), 'pos'])
#         # print(format_sentence(i))
#
# neg = []
# with open("./neg_tweets.txt",'r', encoding='UTF-8') as f:
#     for i in f:
#         neg.append([format_sentence(i), 'neg'])
#
#
# training = pos[:int((.8)*len(pos))]+neg[:int((.8)*len(neg))]
# test = pos[int((.8)*len(pos)):]+neg[int((.8)*len(neg)):]
#
#
# from nltk.classify import NaiveBayesClassifier
# classifier = NaiveBayesClassifier.train(training)
# classifier.show_most_informative_features()
#
#
# example = "Cats are awesome!"
# print(classifier.classify(format_sentence(example)))
#
# example = "I don't like dogs."
# print(classifier.classify(format_sentence(example)))
#
# example = "I don't have headache"
# print(classifier.classify(format_sentence(example)))
#
# example = "Don't thank me!"
# print(classifier.classify(format_sentence(example)))
#
# from nltk.classify.util import accuracy
# print(accuracy(classifier, test))



