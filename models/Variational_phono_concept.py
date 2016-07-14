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

#extract words manually
minLength = 1
words = []
concepts = []
lon = []
lat = []
allConcepts = set()
for lang in languages:
    c = 0
    for w in lang.wordList:
        if len(w) >= minLength:
            words.append(w)
            concepts.append(c)
            allConcepts.add(c)
            lon.append(lang.info["lon"])
            lat.append(lang.info["lat"])
        c += 1

words = Language.extractListOfWords(languages,minLength = 1)
print(len(words))

phonoVectors = np.zeros((len(words),nDim_PhonemeVectors*maxLength))
conceptVectors = np.zeros((len(words),len(allConcepts)),dtype=np.bool)
from utils import getWordMatrix
for i in range(len(words)):
    phonoVectors[i] = getWordMatrix(words[i], model, padToMaxLength=maxLength).flatten()
    conceptVectors[i,concepts[i]] =  True
    


#########
#NETWORK
############
import numpy as np
from keras.layers import Input, Dense, Lambda,merge,Convolution2D,Reshape,MaxPooling2D,UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras import backend as K, objectives
from keras.regularizers import l2



#batch_size = len(phonoVectors.reshape((-1,nDim_phono)))
batch_size = 37
original_dim_phono = nDim_phono
original_dim_concept = len(allConcepts)
latent_dim = 100
intermediate_dim_phono = 500
intermediate_dim_concept = 500
intermediate_dim_concat = 500
epsilon_std = 0.01
nb_epoch = 20
#l2_value = 0.01
l2_value = 0

#encoder phono
input_phono = Input(batch_shape=(batch_size, original_dim_phono))
#encoder phono

h_phono = Dense(intermediate_dim_phono,activation="relu",name="layer_h_phono")(input_phono)

#encoder concepts

input_concept = Input(batch_shape=(batch_size, original_dim_concept))
h_concept = Dense(intermediate_dim_concept,activation="relu",name="layer_h_concept")(input_phono)

h_concat = merge([h_phono,h_concept],mode="concat",name="layer_h_concat")
z_mean = Dense(latent_dim,name="z_mean")(h_concat)
z_log_std = Dense(latent_dim,name="z_log_std")(h_concat)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_std])`
z = Lambda(sampling, output_shape=(latent_dim,),name="layer_z")([z_mean, z_log_std])

#decoder phono
# we instantiate these layers separately so as to reuse them later
phono_decoding_layer_intermediate = Dense(intermediate_dim_phono,activation="relu",name="phono_decoding_layer_intermediate")
phono_decoding_intermediate = phono_decoding_layer_intermediate(z)

phono_decoding_layer_decoded = Dense(original_dim_phono,activation="linear",name="phono_decoding_layer_decoded")
phono_decoded = phono_decoding_layer_decoded(phono_decoding_intermediate)

#decoder concepts

concept_decoding_layer_intermediate = Dense(intermediate_dim_concept,activation="relu",name="concept_decoding_layer_intermediate")
concept_decoding_intermediate = concept_decoding_layer_intermediate(z)

concept_decoding_layer_decoded = Dense(original_dim_concept,activation="sigmoid",name="concept_decoding_layer_decoded")
concept_decoded = concept_decoding_layer_decoded(concept_decoding_intermediate)


def vae_loss(input_phono,phono_decoded):
    mse_loss_phono = objectives.mse(input_phono, phono_decoded)
    ent_loss_concept = objectives.categorical_crossentropy(input_concept, concept_decoded)

    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    return (
             mse_loss_phono 
             + ent_loss_concept
             + kl_loss
             )

vae = Model([input_phono,input_concept], [phono_decoded,concept_decoded])
vae.compile(optimizer='Adam', loss=vae_loss)

vae.fit(x=[phonoVectors.reshape((-1,nDim_phono)),conceptVectors],
         y=[phonoVectors.reshape((-1,nDim_phono)),conceptVectors],
      batch_size=batch_size, nb_epoch=nb_epoch)
encoder = Model( [input_phono,input_concept], z_mean)
#embeddings = encoder.predict(x=[phonoVectors.reshape((-1,nDim_phono))],batch_size=batch_size)
#embeddings = encoder.predict(x=phonoVectors,batch_size=batch_size)
# generator, from latent space to reconstructed inputs


generator_input = Input(shape=(latent_dim,),name="generator_input")
#phono decoding
phono_decoded_intermediate_generator = phono_decoding_layer_intermediate(generator_input)
phono_decoded_generator = phono_decoding_layer_decoded(phono_decoded_intermediate_generator)
#concept decoding
concept_decoded_intermediate_generator = concept_decoding_layer_intermediate(generator_input)
concept_decoded_generator = concept_decoding_layer_decoded(concept_decoded_intermediate_generator)



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
            embeddings.append(encoder.predict([getWordMatrix(word, model, padToMaxLength=maxLength).reshape(1,-1),conceptVectors[c_word].reshape(1,-1)],batch_size=1))
                            
        else:
            embeddings.append(None)
        
        c_word += 1
    lang.info["embeddings"] = embeddings
    c_lang += 1
print("pickling ...")
pickle.dump(languages,open("languages_var_phono_concept.pkl","wb"))
