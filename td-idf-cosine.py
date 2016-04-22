#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

train_set = ["The sky is blue.", "The sun is bright."] #Documents
test_set = ["The sun in the sky is bright."] #Query

while True:
	try:
		stopwords = stopwords.words('english')
		break
	except LookupError as e:
		print e
		nltk.download()

vectorizer = CountVectorizer(stop_words = stopwords)
# print vectorizer

trainVs = vectorizer.fit_transform(train_set).toarray()
testVs = vectorizer.transform(test_set).toarray()
print 'Strings:', [x.encode('ascii','replace') for x in vectorizer.get_feature_names()]
print 'Fit Vectorizer to train set', trainVs
print 'Transform Vectorizer to test set', testVs

for trainV in trainVs:
    print 'trainV:', trainV
    for testV in testVs:
        print 'testV:', testV
        cosine = np.inner(trainV, testV)/(LA.norm(trainV)*LA.norm(testV))
        print 'cosine:', cosine

transformer = TfidfTransformer()
# print transformer

transformer.fit(trainVs)
print
print transformer.transform(trainVs).toarray()

transformer.fit(testVs)
print 
tfidf = transformer.transform(testVs)
print tfidf.todense()
