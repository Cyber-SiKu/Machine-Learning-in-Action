import bayes
from numpy import *
from imp import reload
import re

reload(bayes)

listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)

bayes.setOfWords2Vec(myVocabList,listOPosts[0])
bayes.setOfWords2Vec(myVocabList,listOPosts[3])

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))

p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)

regEx = re.compile('\\W*')
emailText = open('email/ham/6.txt').read()
listOfTokens=regEx.split(emailText)
