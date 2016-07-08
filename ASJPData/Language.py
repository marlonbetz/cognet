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
    @staticmethod
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
                    wordList = data[10:]
                    languages.append(Language(info,wordList))
            c += 1
            
        return languages
    
