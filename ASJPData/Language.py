import codecs
import sys

class Language(object):
    def __init__(self,info,wordList):
        self.info = info 
        self.wordList = wordList
    def __str__(self):
        return "Language{\n\tinfo:"+str(self.info)+"\n\twordList:"+str(self.wordList)+"}"
    def __repr__(self):
        return self.__str__()

def loadLanguagesFromASJP(pathToCorpus="data/dataset.tab",header=True):
    languages = []
    c = 0
    for line in codecs.open(pathToCorpus,"r","utf-8"):
        if header:
            if c > 0 : 
                data = line.split("\t")
                info = dict()
                info["name"] = data[0]
                info["wls_fam"] = data[1]
                info["wls_gen"] = data[2]
                info["e"] = data[3]
                info["hh"] = data[4]
                info["lat"] = data[5]
                info["lon"] = data[6]
                info["pop"] = data[7]
                info["wcode"] = data[8]
                info["iso"] = data[9]
                if "\r\n" in data[-1]: 
                    data[-1] = data[-1][:-2]
                wordList = []
                for d in data[10:]:
                    tmp = d.split(",")[0]
                    #print(tmp)
                    if len(tmp)>0:
                        tmp = tmp.split()[0]
                    wordList.append(tmp)
                languages.append(Language(info,wordList))
        c += 1
            
    return languages

def getListOfLanguagesWithMinimalConceptCount(languages,n):
    
    languages_tmp = []
    for lang in languages:
        if len([True for w in lang.wordList if len(w) > 0]) >= n:
            languages_tmp.append(lang)
    return languages_tmp

def getListOfLanguagesWithoutSpecificInfo(languages,key,value):
    
    languages_tmp = []
    for lang in languages:
        if lang.info[key] != value:
            languages_tmp.append(lang)
    return languages_tmp
    
def extractListOfWords(languages,minLength=1):
    tmp = []
    for lang in languages:
        for w in lang.wordList:
            if len(w) >= minLength:
                tmp.append(w)
    return tmp


def getSetOfUncompleteConcepts(languages):
    tmp = set()
    for i in range(100):
        for lang in languages:
            if len(lang.wordList[i]) < 1:
                print(lang.info["name"],i)
                tmp.add(i)
                break
    return tmp
