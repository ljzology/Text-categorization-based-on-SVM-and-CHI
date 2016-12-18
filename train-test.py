# -*- encoding: utf-8 -*-

import os
import re
from collections import OrderedDict
import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import decomposition
from sklearn.pipeline import FeatureUnion
# import sklearn.model_selection
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
if __name__ == '__main__':
    startTime = time.clock()

    jieba.analyse.set_stop_words('resource/stopwords.txt')  # 去停词路径

    print ('trainset')
    corpus = []
    valueList = []
    fopen = open("resource/user_tag_query.2W.TRAIN", "rb")  # 训练集路径

    for line in fopen:
        if (len(corpus) == 100):
            break
        l = line.split("	", 4)
        if l[2] == '0':
            continue
        ll = jieba.analyse.extract_tags(l[4], 99999)  # 结巴分词去掉stopwords同时进行分词
        # ll = [i for i in ll if i >= u'\u4e00' and i <= u'\u9fa5']#过滤掉非中文
        valueList.append(int(l[2]))
        str_convert = ' '.join(ll)
        # print u' '.join(ll)
        corpus.append(str_convert)
    fopen.close()
    print "len(corpus) = ", len(corpus)
    print "type(corpus) =", type(corpus)
    print " len(valueList) = ", len(valueList)

    # print ('testset')
    # corpus_test = []
    # id = []
    # ftest = open("resource/user_tag_query.2W.TEST", "rb")  # 测试集路径
    # for line in ftest:
    #     # line = line.decode("utf-8")
    #     m += 1
    #     if (m == 5) :
    #         m = 0
    #         break
    #     l = line.split("	", 1)
    #     id.append(l[0])
    #     ll = jieba.analyse.extract_tags(l[1], 20)  # 结巴分词去掉stopwords同时进行分词
    #
    #     # ll = [i for i in ll if i >= u'\u4e00' and i <= u'\u9fa5']#过滤掉非中文
    #
    #     str_convert = ' '.join(ll)
    #     corpus_test.append(str_convert)
    #     # corpus.append(str_convert)
    # ftest.close()

    time1 = time.clock()
    print "jieba take: %f s" % (time1 - startTime)

    from sklearn.cross_validation import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(corpus, valueList, random_state=1, train_size=0.6)
    print "len(x_train) =", len(x_train), type(x_train)
    print "len(y_train) =", len(y_train), type(y_train)
    print "len(x_test) =", len(x_test), type(x_test)
    print "len(y_test) =", len(y_test), type(y_test)
    print 'tfidf'

    # pca = decomposition.PCA(n_components=20000)
    # x_train = pca.fit(x_train)
    # x_test = pca.transform(x_test)

    # print('*************************\nCV\n*************************')
    # cv1 = CountVectorizer(min_df=1)
    # tfidf = cv1.fit_transform(x_train)  # 训练集的tfidf矩阵
    # cv2 = CountVectorizer(vocabulary=cv1.vocabulary_)
    # tfidf_test = cv2.fit_transform(x_test)  # 测试集的tfidf矩阵
    # print tfidf
    # print('*************************\nHV\n*************************')
    # n_feature = 200000
    # hv = HashingVectorizer(n_features=n_feature, non_negative=True)
    # tfidf = hv.fit_transform(x_train)  # 训练集的tfidf矩阵
    # tfidf_test = hv.fit_transform(x_test)  # 测试集的tfidf矩阵
    # print tfidf
    print('*************************\nTV\n*************************')
    tv1 = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    tfidf = tv1.fit_transform(x_train)  # 训练集的tfidf矩阵
    tv2 = TfidfVectorizer(vocabulary=tv1.vocabulary_)
    tfidf_test = tv2.fit_transform(x_test)  # 测试集的tfidf矩阵
    print tfidf.shape, tfidf_test.shape, type(tfidf)
    # print('*************************\nHV+TV\n*************************')
    # hv = HashingVectorizer(n_features=n_feature, non_negative=True)
    # tv1 = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    # train_combined_features = FeatureUnion([('hv', hv), ('tv1', tv1)])
    # tfidf = train_combined_features.fit_transform(x_train)  # 训练集的tfidf矩阵
    # tv2 = TfidfVectorizer(vocabulary=tv1.vocabulary_)
    # test_combined_features = FeatureUnion([('hv', hv), ('tv2', tv2)])
    # tfidf = test_combined_features.fit_transform(x_train)  # 训练集的tfidf矩阵

    # train_data = tfidf
    # test_data = tfidf_test
    # -----CHI-----
    chi_num = 200
    ch2 = SelectKBest(score_func=chi2, k=chi_num)
    train_data = ch2.fit_transform(tfidf, y_train)
    test_data = ch2.transform(tfidf_test)
    print train_data.shape, test_data.shape, type(train_data)
    print np.array(train_data)
    # ---PCA---
    # pca = decomposition.PCA(n_components=100)
    # train_data = pca.fit_transform(train_data)
    # test_data = pca.transform(np.array(test_data))
    #
    # print train_data.shape, test_data.shape, type(train_data)

    time2 = time.clock()
    print "tfidf take: %f s" % (time2 - time1)

    from sklearn import linear_model
    from sklearn.svm import SVC
    from sklearn.grid_search import GridSearchCV
    from sklearn import neighbors
    from scipy import sparse
    import xgboost as xgb
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.linear_model import  LinearRegression, LogisticRegression, LogisticRegressionCV
    # from sklearn.linear_model import linearsvc
    # from sklearn.svm import liblinear
    import scipy.sparse.linalg as lin

    trainset = train_data
    testset = test_data
    # print('*************************\nNB\n*************************')
    # nb_clf = MultinomialNB()
    # # nb_clf = GaussianNB()
    # # nb_clf = BernoulliNB()
    # reg = nb_clf.fit(trainset, y_train)
    # pred = reg.predict(testset)
    # print('*************************\nDT\n*************************')
    # # reg = DecisionTreeRegressor(criterion='mse', max_depth=10)
    # reg = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3)
    # dt = reg.fit(trainset, y_train)
    # pred = dt.predict(testset)
    # pred[pred > 1.5] = 2
    # pred[~(pred > 1.5)] = 1
    # print('*************************\nRF\n*************************')
    # clf = RandomForestClassifier(n_estimators=50, criterion='entropy')
    # rf_clf = clf.fit(trainset, y_train)
    # pred = rf_clf.predict(testset)

    # print('*************************\nXgboost\n*************************')
    # dep = [5, 8, 5]
    # weight = [1, 2, 1]
    # sub = [0.86, 0.8, 0.86]
    # col = [0.68, 0.8, 0.68]
    # cla = [7, 3, 7]
    # tree = [1613, 1598, 1294]
    # train = sparse.csr_matrix([[1, 2, 0], [0, 5, 0]])
    # test = sparse.csr_matrix([[0, 1, 0], [0, 0, 4]])
    # for i in [0]:
    #     param = {"learning_rate": 0.01,
    #              "n_estimators": tree[i],
    #              "max_depth": dep[i],
    #              "min_child_weight": weight[i],
    #              "gamma": 0,
    #              "num_class": 7,
    #              "subsample": sub[i],
    #              "colsample_bytree": col[i],
    #              "objective": 'multi:softmax',
    #              "nthread": 4,
    #              "scale_pos_weight": 1,
    #              "seed": 27}
    #
    #     cf = xgb.train(param, xgb.DMatrix(trainset, y_train))
    #     pred = cf.predict(xgb.DMatrix(testset, missing=0))
    # print('*************************\nSVM\n*************************')
    # print trainset.shape, len(y_train), testset.shape
    # svclf = SVC(kernel='linear')  # default with 'rbf'
    # # c_can = np.logspace(-2,2,10)
    # # svclf = GridSearchCV(svclf,param_grid={'C':c_can}, cv=3)
    # svclf.fit(trainset, y_train)  # 先只预测年龄
    # # print '交叉验证最佳参数： \n', svclf.best_params_
    # pred = svclf.predict(testset)  # 输出预测的年龄数组
    # print('*************************\nKNN\n*************************')
    # print trainset.shape, len(y_train), testset.shape
    # svclf = neighbors.KNeighborsClassifier(n_neighbors=5)
    # svclf.fit(trainset, y_train)  # 先只预测年龄
    # pred = svclf.predict(testset)  # 输出预测的年龄数组
    # print('*************************\nSGD\n*************************')
    # clf = linear_model.SGDClassifier()
    # clf.fit(train_data, y_train)
    # pred = clf.predict(test_data)
    print('*************************\nLineR & LogR & LogRCV\n*************************')
    # clf = LinearRegression()
    # clf = LogisticRegression()
    clf = LogisticRegressionCV()     # cross-validation generator
    clf.fit(train_data, y_train)
    pred = clf.predict(test_data)
    # pred[pred > 1.5] = 2
    # pred[~(pred > 1.5)] = 1
    pred[pred > 5.5] = 6
    pred[4.5 < pred.all() < 5.5] = 5
    pred[3.5 < pred.all() < 4.5] = 4
    pred[2.5 < pred.all() < 3.5] = 3
    pred[1.5 < pred.all() < 2.5] = 2
    pred[pred.all() < 1.5] = 1
    print pred

    match = 0
    for p in zip(pred, y_test):
        if p[0] == p[1]:
            match += 1
    print match, tfidf_test.shape[0]
    print float(match) / tfidf_test.shape[0]
    # fw = open("test.csv", 'w')
    # for (i, la) in zip(id, pred):
    #   fw.write(i + " " + la[0] + '\n')
    # fw.close()

    time3 = time.clock()
    print "idf take: %f s" % (time3 - time2)

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


	from sklearn.svm import SVC
	print('*************************\nSVM\n*************************')
	svclf = SVC(kernel = 'linear')#default with 'rbf'
	svclf.fit(trainMatrix,valueList) #先只预测年龄
	pred = svclf.predict(fea_test);  #输出预测的年龄数组

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
