# Text-categorization-based-on-SVM-and-CHI
My teammate take part the competition holded by CCF, and we get a satisfactory fruit, you can have a look at the website 
http://www.wid.org.cn/data/science/player/competition/detail/description/239

# Data 
if you want the data, you can send email to ljzology@163.com
you can see the illustrate in this URL:
http://www.wid.org.cn/data/science/player/competition/detail/data/239

# Feature extract
I use ifidf to choose the tems, and use chi-square test to choose the feature from tems.

# Algorithm selection
this is an issue of text categorization, the features is very sparse, and the dimension of feature is as big as 960000 
before chi-square.
The svm algorithm perform better than other algorithms when handle sparse matrix, and I have made a seriers of contrast with 
other algorithms, like LR, RF, NB, KNN, XGboost. The experimental result show that SVM is the best in this data.

