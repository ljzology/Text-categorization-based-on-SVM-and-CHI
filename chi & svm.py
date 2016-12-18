# -*- encoding: utf-8 -*-

import os
import re
from collections import OrderedDict
import numpy as np
import jieba
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
from sklearn.feature_selection import SelectKBest, chi2

def show_accuracy(a, b):
    acc = a.ravel() == b.ravel()
    print '正确率：%.2f%%' % (100 * float(acc.sum()) / a.size)

def firstThread():
    print "firstThread"

def agefun(a,b,c,i):
    print "predict ",i ," : --------------->"
    time1 = time.clock()
    svclf = SVC(kernel='linear')  # default with 'rbf'
    svclf.fit(a, b)  # 先只预测年龄
    pred = svclf.predict(c)  # 输出预测的年龄数组

    fw = open("result/result-svm-"+i+"-thread-chi-8.3w-subdf.csv", 'w')
    for (i, la) in zip(id, pred):
        fw.write(i + " " + la + '\n')
    fw.close()
    print "agefun cost time %d s" % (time.clock() - time1)
def genderfun(a,b,c,i):
    print "predict ",i ," : --------------->"
    svclf = SVC(kernel='linear')  # default with 'rbf'
    svclf.fit(a, b)  # 先只预测年龄
    pred = svclf.predict(c)  # 输出预测的年龄数组

    fw = open("result/result-svm-"+i+"-thread-chi-8.3w-subdf.csv", 'w')
    for (i, la) in zip(id, pred):
        fw.write(i + " " + la + '\n')
    fw.close()
def edufun(a,b,c,i):
    print "predict ",i ," : --------------->"
    svclf = SVC(kernel='linear')  # default with 'rbf'
    svclf.fit(a, b)  # 先只预测年龄
    pred = svclf.predict(c)  # 输出预测的年龄数组

    fw = open("result/result-svm-"+i+"-thread-chi-8.3w-subdf.csv", 'w')
    for (i, la) in zip(id, pred):
        fw.write(i + " " + la + '\n')
    fw.close()

if __name__ == '__main__':

    tfidf_topk = 99999
    startTime = time.clock()

    jieba.analyse.set_stop_words('resource/stopwords.txt')  # 去停词路径
    jieba.suggest_freq(('曹云金','郭德纲','林允儿','刘炜','新日'))

    print ('trainset')
    corpus = [[],[],[]]
    valueList = [[],[],[]]
    fopen = open("resource/user_tag_query.10W.TRAIN", "rb")  # 训练集路径

    m = 0
    for line in fopen:
        l = line.split("	", 4)
        # if (m == 1000):
        #     m = 0
        #     break
        # m += 1
        ll = jieba.analyse.extract_tags(l[4], tfidf_topk)  # 结巴分词去掉stopwords同时进行分词
        # ll = [i for i in ll if i >= u'\u4e00' and i <= u'\u9fa5']#过滤掉非中文
        for i in (1,2,3) :
            if (l[i] != '0'):
                valueList[i-1].append(l[i])
                str_convert = ' '.join(ll)
                corpus[i-1].append(str_convert)
    fopen.close()

    print ('testset')
    corpus_test = []
    id = []
    ftest = open("resource/user_tag_query.10W.TEST", "rb")  # 测试集路径
    for line in ftest:
        # if (m == 50):
        #     m = 0
        #     break
        # m += 1
        l = line.split("	", 1)
        id.append(l[0])
        ll = jieba.analyse.extract_tags(l[1], tfidf_topk)  # 结巴分词去掉stopwords同时进行分词
        # ll = [i for i in ll if i >= u'\u4e00' and i <= u'\u9fa5']#过滤掉非中文

        str_convert = ' '.join(ll)
        corpus_test.append(str_convert)
    ftest.close()

    time1 = time.clock()
    print "jieba take: %d s" % (time1 - startTime)

    print 'tfidf'
    tfidf = [[],[],[]]
    tfidf_test = [[],[],[]]

    ch2 = SelectKBest(score_func=chi2, k=83000)

    for i in (0,1,2):
        tv1 = TfidfVectorizer()    #sublinear_tf=True, max_df=0.5
        # print "corpus[%d].shape = " % i, corpus[i].shape
        tfidf[i] = tv1.fit_transform(corpus[i])  # 训练集的tfidf矩阵
        print "before tfidf[%d].shape = " % i, tfidf[i].shape
        tfidf[i] = ch2.fit_transform(tfidf[i], valueList[i])
        print "after tfidf[%d].shape = " % i, tfidf[i].shape

        tv11 = TfidfVectorizer(vocabulary=tv1.vocabulary_)
        # print "corpus_test[%d].shape = " % i, corpus_test[i].shape
        tfidf_test[i] = tv11.fit_transform(corpus_test)  # 测试集的tfidf矩阵
        print "before tfidf_test[%d].shape = " % i, tfidf_test[i].shape
        tfidf_test[i] = ch2.transform(tfidf_test[i])
        print "after tfidf_test[%d].shape = " % i, tfidf_test[i].shape


    time2 = time.clock()
    print "idf take: %d s" % (time2 - time1)

    from sklearn.svm import SVC
    import scipy.sparse.linalg as lin
    import threading

    print('*************************\nSVM\n*************************')
    threads = []

    t0 = threading.Thread(target= firstThread)
    threads.append(t0)
    t1 = threading.Thread(target= agefun, args=(tfidf[0], valueList[0], tfidf_test[0], "age"))
    threads.append(t1)
    t2 = threading.Thread(target= genderfun, args=(tfidf[1], valueList[1], tfidf_test[1], "gender"))
    threads.append(t2)
    t3 = threading.Thread(target= edufun, args=(tfidf[2], valueList[2], tfidf_test[2], "edu"))
    threads.append(t3)

    t1.start()
    t2.start()
    t3.start()
    # for t in threads:
    #     t.setDaemon(True)  #shou hu jin cheng
    #     t.start()
    # t.join()

    # time3 = time.clock()
    # print "idf take: %f s" % (time3 - time2)

'''
	model = Word2Vec.load('D:/BaiduYunDownload/word2vector.model')#载入本地的词向量模型 这个我还没跑 有问题

	wordMatrix = csr_matrix((len(word),len(word)))#聚类用的单词矩阵

	for i in range(0,len(word)):
		for j in range(0,len(word)):
			wordMatrix[i,j] = model.similarity(word[i],word[j])

 	聚类算法
	clf = KMeans(n_clusters=cluster_num)
	s = clf.fit(wordMatrix)#聚类
	joblib.dump(clf, "D:/BaiduYunDownload/cluster_model.m")#将聚类模型保存到本地

	分类算法
	trainMatrix = csr_matrix((tfidf.shape[0],cluster_num))#训练分类输入数据 样本为训练用户数，维度为聚类数
	for i in range(0,tfidf.shape[0]):
		for j in range(0,len(word)):
			trainMatrix[i,clf.labels_[j]] = train[i,clf.labels_[j]] + tfidf[i,j]

	fea_test = csr_matrix((tfidf_test.shape[0],cluster_num))#测试分类输入数据 样本为测试用户数，维度为聚类数
	for i in range(0,tfidf_test.shape[0]):
		for j in range(0,len(word)):
			fea_test[i,clf.labels_[j]] = train[i,clf.labels_[j]] + tfidf_test[i,j]


	'''

'''
	from sklearn.cluster import KMeans
	trainData = up.zeros([len(word),100])

	clf = KMeans(n_clusters=100)
	s = clf.fit(weight)
	print s
	#20个中心点
	print(clf.cluster_centers_)

	#每个样本所属的簇
	print(clf.labels_)
	i = 0
	while i <= len(clf.labels_):
		print i, clf.labels_[i]
		i = i + 1

	for i in range(weight.shape[0]):
		for j in range(weight.shape[1]):
			trainData[i,]

	trainData =


		print(user_id)
		print(age)
		print(gender)
		print(education)
		print(ll)
		'''




