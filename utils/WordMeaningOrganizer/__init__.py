import numpy as np
import regex
from keras.preprocessing.sequence import pad_sequences

def getWordMatrix(word,model):
    phonemes_alone="pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieaouE3"
    phonemeSearchRegex = "["+phonemes_alone+"][\"\*]?(?!["+phonemes_alone+"]~|["+phonemes_alone+"]{2}\$)|["+phonemes_alone+"]{2}?~|["+phonemes_alone+"]{3}?\$"
    phonemes = regex.findall(phonemeSearchRegex, word)
    wordVector = []
    for phoneme in phonemes:
        #if phoneme not in model, get single chars as phonemes instead
        if phoneme not in model:
            for ph in regex.findall(phonemes_alone, phoneme):
                wordVector.append(model[ph])
        else:       
            wordVector.append(model[phoneme])    
    return wordVector
def getWordMeaningMatrices(dictionary,model,languages = None,maxLength=20,boundary=None):
    words = []
    if boundary:
        if boundary in model:
            if languages:
                    for lang in languages: 
                        tmp = []
                        for i_meaning in range(len(dictionary[lang])):
                            if len(dictionary[lang][i_meaning][2:-1]) > 0:
                                tmp.append(boundary+dictionary[lang][i_meaning][2:-1]+boundary)
                            else:
                                tmp.append(dictionary[lang][i_meaning][2:-1])
                                
                        words.append(tmp)
            else:  
                for lang in dictionary: 
                    tmp = []
                    for i_meaning in range(len(dictionary[lang])):
                        if len(dictionary[lang][i_meaning][2:-1]) > 0:
                            tmp.append(boundary+dictionary[lang][i_meaning][2:-1]+boundary)
                        else:
                            tmp.append(dictionary[lang][i_meaning][2:-1])
                    words.append(tmp)
        else:
            if languages:
                for lang in languages: 
                    tmp = []
                    for i_meaning in range(len(dictionary[lang])):
                        tmp.append(dictionary[lang][i_meaning][2:-1])
                    words.append(tmp)
            else:  
                for lang in dictionary: 
                    tmp = []
                    for i_meaning in range(len(dictionary[lang])):
                        tmp.append(dictionary[lang][i_meaning][2:-1])
                    words.append(tmp)
    else:
        if languages:
                for lang in languages: 
                    tmp = []
                    for i_meaning in range(len(dictionary[lang])):
                        tmp.append(dictionary[lang][i_meaning][2:-1])
                    words.append(tmp)
        else:  
            for lang in dictionary: 
                tmp = []
                for i_meaning in range(len(dictionary[lang])):
                    tmp.append(dictionary[lang][i_meaning][2:-1])
                words.append(tmp)
    words = np.array(words)
    words = words.transpose()
    
    #check that only meanings are used where all languages have examples
    meanings_toScrap = set()
    for i_meaning in range(len(words)):
        for i_lang in range(len(words[i_meaning])):
            if len(words[i_meaning,i_lang]) < 1 :
                meanings_toScrap.add(i_meaning)
    print(meanings_toScrap)
    words_tmp = []
    for i_meaning in range(len(words)):
        if i_meaning not in meanings_toScrap:
            meaning_tmp = []
            for i_lang in range(len(words[i_meaning])):
                meaning_tmp.append(words[i_meaning,i_lang])
            words_tmp.append(meaning_tmp)
    words = np.array(words_tmp)
               

                
    n_meanings = words.shape[0]
    n_langs =  words.shape[1]
    meanings = np.zeros((words.shape[0],words.shape[1],n_meanings),dtype=np.bool)
    for m in range(n_meanings):
        for l in range(n_langs):
            meanings[m,l,m] = True
            
    #word vectors
    nDim_PhonemeVectors = len(model[list(model.vocab.keys())[0]])
    nDim_phono = maxLength * nDim_PhonemeVectors
    wordVectors = np.zeros((words.shape[0],words.shape[1],nDim_phono))
    for meaning in range(n_meanings):
        print("processing meaning",meaning,"/",n_meanings)
        for lang in range(n_langs):
            if len(getWordMatrix(words[meaning][lang], model)) < 1:
                wordVectors[meaning,lang] = np.zeros(nDim_phono)
            else:
                wordVectors[meaning,lang] = pad_sequences(sequences=np.array(getWordMatrix(words[meaning][lang], model),dtype = "float32").reshape(1,-1,nDim_PhonemeVectors),
                                                       maxlen=maxLength,
                                                       padding="post", truncating="post",dtype="float32").flatten()

 
    return words,wordVectors,meanings

    
# from gensim.models.word2vec import *
# import pickle
# fname = "/Users/marlon/Documents/pythonWorkspace/KerasSandbox/KerasSandbox/GensimSkipGramNegativeSamplingVectors/model.pkl"
# model = Word2Vec.load(fname)
# pathDictionary = "/Users/marlon/Documents/pythonWorkspace/KerasSandbox/KerasSandbox/Resources/languages.pkl"
# dictionary = pickle.load(open(pathDictionary , "rb" ) )
# words,wordVectors,meanings = getWordMeaningMatrices(dictionary=dictionary,languages=["POLISH","RUSSIAN"],model=model)
# print(words.shape,wordVectors.shape,meanings.shape)
# print(words[0,0],len(words[0,0]))