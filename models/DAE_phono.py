import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import gensim
import regex
from sklearn import preprocessing
import skimage.transform as skt
import sys
import pandas
import h5py
from keras.preprocessing.sequence import pad_sequences


#########
#PREPROCESSING
############
from gensim.models.word2vec import *
import pickle
import ASJPData

lang_names = {"POLISH","KASHUBIAN","RUSSIAN","UKRAINIAN","CZECH","SLOVAK","CROATIAN","BULGARIAN","SLOVENIAN","BOSNIAN",
             "LOWER_SORBIAN","UPPER_SORBIAN",
             "LATVIAN","LITHUANIAN",
             "TURKISH","AZERBAIJANI_NORTH","UZBEK",
             "STANDARD_GERMAN","ENGLISH","DUTCH","EASTERN_FRISIAN","FRISIAN_WESTERN","AFRIKAANS",
             "DANISH","SWEDISH","NORWEGIAN_BOKMAAL","NORWEGIAN_NYNORSK_TOTEN","NORWEGIAN_RIKSMAL",
             "ICELANDIC","FAROESE",
             "ITALIAN","ROMANSH_SURSILVAN","FRENCH","OCCITAN_ARANESE","CATALAN","BALEAR_CATALAN",
             "SPANISH","PORTUGUESE","ROMANIAN",
             "WELSH","BRETON","CORNISH","GAELIC_SCOTTISH","IRISH_GAELIC",
             "GREEK",
             "EASTERN_ARMENIAN","WESTERN_ARMENIAN",
             "BASQUE",
             "HUNGARIAN","FINNISH","ESTONIAN","KOMI_PERMYAK","KOMY_ZYRIAN"
             }


fname = "/Users/marlon/Documents/pythonWorkspace/KerasSandbox/KerasSandbox/GensimSkipGramNegativeSamplingVectors/model.pkl"
model = Word2Vec.load(fname)

maxLength = 20

nDim_PhonemeVectors = 200
nDim_phono = nDim_PhonemeVectors*maxLength

import ASJPData.Language as Language
languages = Language.loadLanguagesFromASJP("/Users/marlon/Documents/pythonWorkspace/cognet/ASJPData/data/dataset.tab")

languages = Language.getListOfLanguagesWithoutSpecificInfo(languages,"wls_gen","ARTIFICIAL")

languages = [language for language in languages if language.info["name"] in lang_names]
words = Language.extractListOfWords(languages,minLength = 1)
print(len(words))

wordVectors = np.zeros((len(words),nDim_PhonemeVectors*maxLength))
from utils import getWordMatrix
for i in range(len(words)):
    wordVectors[i] = getWordMatrix(words[i], model, padToMaxLength=maxLength).flatten()



#########
#NETWORK
############
import numpy as np
from keras.layers import Input, Dense, Lambda,merge,Convolution2D,Reshape,MaxPooling2D,UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras import backend as K, objectives
from keras.regularizers import l2



batch_size = len(wordVectors.reshape((-1,nDim_phono)))
batch_size = 50
original_dim_phono = nDim_phono
#original_dim_meaning = n_meanings
#outputChannels = 5
latent_dim = 500
intermediate_dim_phono = 2000
intermediate_dim_meaning = 128
intermediate_dim_concat = 128
epsilon_std = 0.1
nb_epoch = 10
l2_value = 0.01
#l2_value = 0

input_phono = Input(batch_shape=(batch_size, original_dim_phono))
input_phono_corrupted= GaussianNoise(sigma=0.1,name="GaussianNoise")(input_phono)

h_phono = Dense(latent_dim,activation="tanh")(input_phono_corrupted)

y_predicted_layer = Dense(original_dim_phono,activation="linear",name="y_predicted_layer")

decoded_phono = y_predicted_layer(h_phono)




vae = Model([input_phono], [decoded_phono])
vae.compile(optimizer='Adam', loss="mse")

vae.fit(x=[wordVectors.reshape((-1,nDim_phono))],
         y=[wordVectors.reshape((-1,nDim_phono))],
      batch_size=batch_size, nb_epoch=nb_epoch)
encoder = Model( [input_phono], h_phono)


print("creating embeddings ...")
c_lang = 0
for lang in languages:
    print(c_lang,"/",len(languages),lang.info["name"])
    embeddings = []
    c_word = 0
    for word in lang.wordList:
        if len(word) > 0:
            embeddings.append(encoder.predict(getWordMatrix(word, model, padToMaxLength=maxLength).reshape(1,-1),batch_size=1))
                            
        else:
            embeddings.append(None)
        
        c_word += 1
    lang.info["embeddings"] = embeddings
    c_lang += 1
print("pickling ...")
pickle.dump(languages,open("languages_dae.pkl","wb"))


