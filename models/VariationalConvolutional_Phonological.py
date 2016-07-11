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
#fname = "gensimWord2vecPhonemeVectors_150_20d_window2_PhonemeLength3.pkl"
model = Word2Vec.load(fname)
# pathDictionary = "/Users/marlon/Documents/pythonWorkspace/KerasSandbox/KerasSandbox/Resources/languages.pkl"
# dictionary = pickle.load(open(pathDictionary , "rb" ) )
# languages = ["POLISH"
#              ,"RUSSIAN","UKRAINIAN","CZECH","SLOVAK","CROATIAN","BULGARIAN","SLOVENIAN","BOSNIAN",
#              "LATVIAN","LITHUANIAN",
#              "TURKISH","AZERBAIJANI_NORTH","UZBEK",
#              "STANDARD_GERMAN","ENGLISH","DUTCH","EASTERN_FRISIAN","FRISIAN_WESTERN","AFRIKAANS",
#              "DANISH","SWEDISH","NORWEGIAN_BOKMAAL","NORWEGIAN_NYNORSK_TOTEN","NORWEGIAN_RIKSMAL",
#              "ICELANDIC","FAROESE",
#              "ITALIAN","ROMANSH_SURSILVAN","FRENCH","OCCITAN_ARANESE","CATALAN","BALEAR_CATALAN",
#              "SPANISH","PORTUGUESE","ROMANIAN",
#              "WELSH","BRETON","CORNISH","GAELIC_SCOTTISH","IRISH_GAELIC",
#              "BASQUE",
#              "HUNGARIAN","FINNISH","ESTONIAN"
#              ]
maxLength = 20
# from utils import WordMeaningOrganizer
# words,wordVectors,meanings = WordMeaningOrganizer.getWordMeaningMatrices(dictionary=dictionary,
#                                                     languages=languages,
#                                                     model=model,
#                                                     maxLength=maxLength
#                                                     ,boundary="#")
# print(words.shape,wordVectors.shape,meanings.shape)
nDim_PhonemeVectors = 200
nDim_phono = nDim_PhonemeVectors*maxLength
# nDim_phono = wordVectors.shape[2]
# n_meanings = meanings.shape[2]
# n_langs = wordVectors.shape[1]
# nDim_PhonemeVectors = int(wordVectors.shape[2]/maxLength)
# min_max_scaler = preprocessing.MinMaxScaler()
# wordVectors = np.reshape(min_max_scaler.fit_transform(np.reshape(wordVectors,(-1,nDim_phono))),(n_meanings,n_langs,-1))

import ASJPData.Language as Language
languages = Language.loadLanguagesFromASJP("/Users/marlon/Documents/pythonWorkspace/cognet/ASJPData/data/dataset.tab")

languages = Language.getListOfLanguagesWithoutSpecificInfo(languages,"wls_gen","ARTIFICIAL")
words = Language.extractListOfWords(languages,minLength = 1)
print(len(words))

wordVectors = np.zeros((len(words),nDim_PhonemeVectors*maxLength))
from utils import getWordMatrix
for i in range(len(words)):
    wordVectors[i] = getWordMatrix(words[i], model, padToMaxLength=maxLength).flatten()
#scaler = preprocessing.StandardScaler().fit(wordVectors)
#wordVectors = scaler.transforrm(wordVectors)
    #     for word in language.wordList:
#         if len(word) > 0:
#             print(word)
#             print(getWordMatrix(word, model, padToMaxLength=20).shape)

#sys.exit()



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
batch_size = 389
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


z_mean = Dense(latent_dim)(h_phono)
z_log_std = Dense(latent_dim)(h_phono)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])

# we instantiate these layers separately so as to reuse them later
y_merged_layer = Dense(32*5*50,activation="relu")
y_merged = y_merged_layer(z)
#phono decoding
y_phono_reshaper = Reshape((32,5,50))
y_phono_reshaped = y_phono_reshaper(y_merged)
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


def vae_loss(x_phono,decoded_phono):
    mse_loss_phono = objectives.mse(x_phono, decoded_phono)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return (
             mse_loss_phono + kl_loss)

vae = Model([input_phono], [decoded_phono])
vae.compile(optimizer='Adam', loss=vae_loss)

vae.fit(x=[wordVectors.reshape((-1,nDim_phono))],
         y=[wordVectors.reshape((-1,nDim_phono))],
      batch_size=batch_size, nb_epoch=nb_epoch)
encoder = Model( [input_phono], z_mean)
#embeddings = encoder.predict(x=[wordVectors.reshape((-1,nDim_phono))],batch_size=batch_size)
#embeddings = encoder.predict(x=wordVectors,batch_size=batch_size)
# generator, from latent space to reconstructed inputs


generator_input = Input(shape=(latent_dim,),name="generator_input")
#y_merged_layer = Dense(32*5*5,activation="relu")
y_merged_generator = y_merged_layer(generator_input)
#phono decoding
y_phono_reshaped_generator = y_phono_reshaper(y_merged_generator)
y_phono1_generator = conv1_decoder(y_phono_reshaped_generator)
y_phono2_generator = upSampling1_decoder(y_phono1_generator)
y_phono3_generator = conv2_decoder(y_phono2_generator)
y_phono4_generator = upSampling2_decoder(y_phono3_generator)
y_phono5_generator = conv3_decoder(y_phono4_generator)
decoded_phono_generator = y_phono5_reshaper(y_phono5_generator)

generator_phono = Model(generator_input, decoded_phono_generator)


from utils.WordReconstructor import *

#wr_eucl = WordReconstructor(X=[model[w] for w in list(model.vocab.keys())],y=list(model.vocab.keys()),metric="euclidean")
#wr_cos = WordReconstructor(X=[model[w] for w in list(model.vocab.keys())],y=list(model.vocab.keys()),metric="cosine")

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
pickle.dump(languages,open("languages_varconv.pkl","wb"))


# 
# 
# for word,code in zip(words.flatten()[:],embeddings[:]):
#     #code = embeddings[0]
#     #print(code)
#     decoded_p = generator_phono.predict(code.reshape(-1,latent_dim))
#     decoded_p=min_max_scaler.inverse_transform(decoded_p)
# 
#     #decoded_p = min_max_scaler.inverse_transform(decoded_p)
#     decoded_p = decoded_p.reshape((maxLength,nDim_PhonemeVectors))
#     decoded_p_eucl = wr_eucl.reconstruct(decoded_p)
#     decoded_p_cos = wr_cos.reconstruct(decoded_p)
#     decoded_p_eucl = "".join([d[0] for d in decoded_p_eucl])
#     decoded_p_cos = "".join([d[0] for d in decoded_p_cos])
#     
#     #decoded_m = generator_meaning.predict(code.reshape(-1,latent_dim))
#     print(word,decoded_p_eucl,decoded_p_cos)
#     #print(decoded_p)
#     #print(decoded_m)
#     
# print("linear transformations ...")
# from utils import vectorLinspace
# import codecs
# f = codecs.open("linear_transformations.txt","w")
# for word1,code1 in zip(words.flatten()[80:100],embeddings[80:100]):
#     f.write("--------\n")
#     for word2,code2 in zip(words.flatten()[80:100],embeddings[80:100]):
#         f.write("---\n")
# 
#         if word1 != word2:
#             
#             for code_tmp in vectorLinspace(start=code1.flatten(), stop=code2.flatten(), num=10):
#                 
#                 decoded_p = generator_phono.predict(code_tmp.reshape(-1,latent_dim))
#                 decoded_p=min_max_scaler.inverse_transform(decoded_p)
#             
#                 #decoded_p = min_max_scaler.inverse_transform(decoded_p)
#                 decoded_p = decoded_p.reshape((maxLength,nDim_PhonemeVectors))
#                 decoded_p_eucl = wr_eucl.reconstruct(decoded_p)
#                 decoded_p_cos = wr_cos.reconstruct(decoded_p)
#                 decoded_p_eucl = "".join([d[0] for d in decoded_p_eucl])
#                 decoded_p_cos = "".join([d[0] for d in decoded_p_cos])
#                 
#                 f.write(word1+" "+word2+" "+str(decoded_p_eucl)+" "+str(decoded_p_cos)+"\n")
#                 
# pandas.DataFrame(embeddings.transpose(),columns=words.flatten()).to_csv("CogNetEmbeddings.csv")
# 
# sys.exit()
# generator_phono = Model(generator_input, decoded_phono_generator)
