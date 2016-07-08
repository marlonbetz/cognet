import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class WordReconstructor(object):
    def __init__(self,X,y,metric):
        X = np.array(X)
        y = np.array(y)
        X = np.append(X, np.zeros((1,len(X[0]))), axis=0)
        y = np.append(y, np.array(["_"]), axis=0)
        self.classifier = KNeighborsClassifier(n_neighbors=1,metric=metric,algorithm="brute")
        
        self.classifier.fit(X,y)
    def reconstruct(self,matrix):
        s = []
        for vector in matrix:

            s.append(list(self.classifier.predict([vector])))
        return s

