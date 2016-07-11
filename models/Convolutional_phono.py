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
fname = "/Users/marlon/Documents/pythonWorkspace/KerasSandbox/KerasSandbox/GensimSkipGramNegativeSamplingVectors/model.pkl"
model = Word2Vec.load(fname)
#
maxLength = 20

nDim_PhonemeVectors = 200
nDim_phono = nDim_PhonemeVectors*maxLength

import ASJPData.Language as Language
languages = Language.loadLanguagesFromASJP("/Users/marlon/Documents/pythonWorkspace/cognet/ASJPData/data/dataset.tab")

languages = Language.getListOfLanguagesWithoutSpecificInfo(languages,"wls_gen","ARTIFICIAL")
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
latent_dim = 500
intermediate_dim_phono = 128
intermediate_dim_meaning = 128
intermediate_dim_concat = 128
epsilon_std = 0.01
nb_epoch = 1
#l2_value = 0.01
l2_value = 0

input_phono = Input(batch_shape=(batch_size, original_dim_phono))
#input_phono_corrupted= GaussianNoise(sigma=0.1,name="GaussianNoise")(input_phono)
#encoder phono
input_phono_img = Reshape((1,maxLength,nDim_PhonemeVectors))(input_phono)
x_phono = Convolution2D(64,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))(input_phono_img)
x_phono = MaxPooling2D((2,2))(x_phono)
x_phono = Convolution2D(32,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))(x_phono)
x_phono = MaxPooling2D((2,2))(x_phono)
h_phono = Reshape((32*5*50,))(x_phono)

y_phono_reshaper = Reshape((32,5,50))
y_phono_reshaped = y_phono_reshaper(h_phono)
conv1_decoder = Convolution2D(32,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))
y_phono1 = conv1_decoder(y_phono_reshaped)
upSampling1_decoder = UpSampling2D((2,2))
y_phono2 = upSampling1_decoder(y_phono1)
conv2_decoder = Convolution2D(64,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))
y_phono3 = conv2_decoder(y_phono2)
upSampling2_decoder = UpSampling2D((2,2))
y_phono4 = upSampling2_decoder(y_phono3)
conv3_decoder = Convolution2D(1,3,3,activation="relu",border_mode="same",W_regularizer=l2(l2_value))
y_phono5 = conv3_decoder(y_phono4)
y_phono5_reshaper =  Reshape((nDim_phono,),name="decoded_phono")
decoded_phono = y_phono5_reshaper(y_phono5)




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
    newpath = "/Users/marlon/Documents/pythonWorkspace/cognet/models/pickled_languages/languages_conv_phono"+str(latent_dim)+"_"+str(nb_epoch)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    #pickle.dump(lang,open("/Users/marlon/Documents/pythonWorkspace/cognet/models/pickled_languages/languages_conv_phono"+str(nb_epoch)+"/"+lang.info["name"]+".pkl","wb"))
    with open("/Users/marlon/Documents/pythonWorkspace/cognet/models/pickled_languages/languages_conv_phono"+str(latent_dim)+"_"+str(nb_epoch)+"/"+lang.info["name"]+".pkl","wb") as f:
        pickle.dump(lang, f)

    c_lang += 1
#print("pickling ...")
#pickle.dump(languages,open("languages_conv.pkl","wb"))
