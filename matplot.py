# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
import matplotlib
import matplotlib.pyplot as plot
from pandas import Series

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def drawFig1():
    a = plot.subplot(1, 1, 1)
    cvList = [0.5417, 0.76417, 0.55817, 0.62134]
    hvList = [0.55617, 0.79433, 0.57233, 0.64094]
    tvList = [0.57017, 0.80950, 0.58933, 0.65633]

    feature1 = [10, 20, 30, 40]
    feature2 = [12, 22, 32, 42]
    feature3 = [14, 24, 34, 44]

    attr = [u'年龄', u'性别', u'学历', u'均值']
    plot.bar(feature1, cvList, facecolor='red', width=2, label='CV', alpha=0.8)
    plot.bar(feature2, hvList, facecolor='green', width=2, label='HV', alpha=0.8)
    plot.bar(feature3, tvList, facecolor='blue', width=2, label='TV', alpha=0.7)

    plot.xticks([13, 23, 33, 43], attr, fontsize =20)
    plot.yticks(fontsize=20)
    plot.ylabel(u'准确率', fontsize=20)

    plot.ylim(0.4, 1)
    plot.xlim(7, 49)
    plot.legend(fontsize=20)    #更改标注的属性
    plot.grid()
    plot.savefig("cv-hv-tv.png")
    plot.show()


def drawFig2():
    x = [8000, 10000, 20000, 40000, 60000, 80000, 100000, 110000, 120000, 130000, 140000, 160000]
    tv = [0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950, 0.80950,
          0.80950]
    tvChi = [0.81383, 0.81533, 0.8156, 0.81383, 0.814, 0.81417, 0.813, 0.81533, 0.81517, 0.81483, 0.81467, 0.814]
    hvChi = [0.805, 0.8075, 0.80683, 0.80683, 0.80683, 0.80717, 0.8065, 0.8055, 0.80517, 0.803, 0.80217, 0.80217]
    cvChi = [0.6955, 0.69868, 0.7145, 0.73967, 0.74183, 0.74333, 0.74717, 0.74483, 0.74517, 0.74617, 0.74733, 0.749]

    x2 = [14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000]
    tvChi2 = [0.81483, 0.81483, 0.815, 0.8156, 0.81583, 0.81617, 0.8155, 0.816, 0.81633, 0.8155, 0.81433, 0.814,
              0.81367]
    # plot.plot(x, tv, 'md-', label='TV', linewidth=2)
    # plot.plot(x, tvChi, 'ro-', label='TV+CHI', linewidth=1.5)
    # plot.plot(x, hvChi, 'g^-', label='HV+CHI', linewidth=1.5)
    # plot.plot(x, cvChi, 'b*-', label='CV+CHI', linewidth=1.5)

    plot.plot(x2, tvChi2, 'rd-', label='TV+CHI', linewidth=3, ms=20)
    plot.plot(x2, tv[:len(x2)], 'ko-', label='TV', linewidth=3, ms=20)
    plot.xlim(12000, 40000)
    plot.ylim(0.806, 0.818)
    plot.xticks(fontsize=20)
    plot.yticks(fontsize=20)


    # plot.xlim(5000,170000)
    plot.legend(loc='lower right', fontsize=20)
    plot.xlabel(u'特征词维度', fontsize=20)
    plot.ylabel(u'准确率', fontsize=20)
    plot.grid()
    # plot.savefig('tv pk tv+chi.png')
    plot.show()


def drawFig3():
    x = [14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000]
    svmd = [0.81483, 0.81483, 0.815, 0.8156, 0.81583, 0.81617, 0.8155, 0.816, 0.81633, 0.8155, 0.81433, 0.814,
            0.81367]
    nbd = [0.78817, 0.7855, 0.78433, 0.78067, 0.77933, 0.77783, 0.777, 0.77667, 0.77167, 0.76933, 0.76617, 0.76217,
           0.76017]
    rfd = [0.7825, 0.7815, 0.78833, 0.7825, 0.77733, 0.77933, 0.77967, 0.78233, 0.78167, 0.77733, 0.77167, 0.78083,
           0.78]
    xgd = [0.81433, 0.81483, 0.81267, 0.81383, 0.81367, 0.81417, 0.8155, 0.81316, 0.81417, 0.81317, 0.81133, 0.81417,
           0.81133]
    LogRd = [0.79483, 0.796, 0.79583, 0.79617, 0.79617, 0.798, 0.79733, 0.79967, 0.79817, 0.79967, 0.79917, 0.7995,
             0.79767]
    SGDd = [0.815, 0.81467, 0.8145, 0.81567, 0.8165, 0.8175, 0.817, 0.81483, 0.8175, 0.81567, 0.81533, 0.81333, 0.813]
    DTd = [0.66717, 0.6695, 0.67383, 0.66767, 0.6655, 0.675, 0.674, 0.667167, 0.6685, 0.66133, 0.6685, 0.6665, 0.669]
    knnd = [0.58, 0.58017, 0.58017, 0.57983, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58033, 0.58033]
    # xgd = [0.81633, 0.81683, 0.81467, 0.81583, 0.81567, 0.81617, 0.8175, 0.81516, 0.81617, 0.81517, 0.81333, 0.81617,
    #        0.81333]
    svmAve = [0.635886667, 0.640443333, 0.643166667, 0.644753333, 0.646333333, 0.64867, 0.649166667, 0.648943333,
              0.650723333, 0.652333333, 0.65311, 0.65431, 0.653613333]
    nbAve = [0.563613333, 0.56489, 0.565943333, 0.566276667, 0.565723333, 0.565, 0.564556667, 0.575946667, 0.562946667,
             0.561443333, 0.560056667, 0.558166667, 0.557823333]
    rfAve = [0.610386667, 0.606113333, 0.609666667, 0.6095, 0.606056667, 0.601776667, 0.6045, 0.60483, 0.6035, 0.601496667, 0.5995,
             0.601833333, 0.60322]
    xgAve = [0.60222, 0.603386667, 0.599546667, 0.599553333, 0.60328, 0.604833333, 0.60289, 0.604776667, 0.60378, 0.606223333, 0.605,
             0.60211, 0.603333333]
    lrAve = [0.639556667, 0.637276667, 0.632333333, 0.636503333, 0.631836667, 0.63211, 0.629443333, 0.632836667, 0.630223333, 0.629556667, 0.630223333,
             0.633056667, 0.630276667]

    plot.plot(x, svmAve, 'ro-', label='SVM', linewidth=3, ms = 8, markersize=12)
    plot.plot(x, nbAve, 'g^-', label='NB', linewidth=3, ms = 8)
    plot.plot(x, rfAve, 'b*-', label='RF', linewidth=3, ms = 10)
    plot.plot(x, xgAve, 'md-', label='KNN', linewidth=3, ms = 8)
    plot.plot(x, lrAve, 'cp-', label='LR', linewidth=3, ms = 8)
    # plot.plot(x, knnd, 'cp-', label='KNN', linewidth=3)
    # plot.plot(x, LogRd, 'yd-', label='LogR', linewidth=3)
    # plot.plot(x, SGDd, 'bp-', label='SGD', linewidth=3)

    plot.xlim(13000, 39000)
    plot.ylim(0.48, 0.67)
    plot.xticks(fontsize=21)
    # plot.xlim(5000,170000)
    plot.legend(loc='lower right')
    plot.xlabel(u'特征词维度', fontsize=20)
    plot.ylabel(u'准确率')
    plot.grid()
    # plot.savefig('svm nb rf xg ave PK.png')
    plot.show()


if __name__ == "__main__":
    # drawFig1()
    # drawFig2()
    drawFig3()
