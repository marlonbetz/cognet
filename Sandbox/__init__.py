import numpy as np 
import pickle 
from ASJPData.Language import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from os import listdir
import sys
languages = []
print("loading language data ...")
#for f_name in listdir(path="/Users/marlon/Documents/pythonWorkspace/cognet/models/pickled_languages/languages_conv_phono500_1"):
#    languages.append(pickle.load(open("/Users/marlon/Documents/pythonWorkspace/cognet/models/pickled_languages/languages_conv_phono500_1/"+f_name,"rb")))
languages = pickle.load(open("/Users/marlon/Documents/pythonWorkspace/cognet/models/languages_varconv_phono.pkl","rb"))

embeddings = []
words = []
concepts = []
names_langs_used =[]
lang_names = {"POLISH"
             ,"RUSSIAN","UKRAINIAN","CZECH","SLOVAK","CROATIAN","BULGARIAN","SLOVENIAN","BOSNIAN",
             "LATVIAN","LITHUANIAN",
             "TURKISH","AZERBAIJANI_NORTH","UZBEK",
             "STANDARD_GERMAN","ENGLISH","DUTCH","EASTERN_FRISIAN","FRISIAN_WESTERN","AFRIKAANS",
             "DANISH","SWEDISH","NORWEGIAN_BOKMAAL","NORWEGIAN_NYNORSK_TOTEN","NORWEGIAN_RIKSMAL",
             "ICELANDIC","FAROESE",
             "ITALIAN","ROMANSH_SURSILVAN","FRENCH","OCCITAN_ARANESE","CATALAN","BALEAR_CATALAN",
             "SPANISH","PORTUGUESE","ROMANIAN",
             "WELSH","BRETON","CORNISH","GAELIC_SCOTTISH","IRISH_GAELIC",
             "BASQUE",
             "HUNGARIAN","FINNISH","ESTONIAN"
             }
print("extracting data ...")
concept_to_plot = 56
for lang in languages:
    #if lang.info["name"] in lang_names:
    embeddings.extend([lang.info["embeddings"][i] for i in range(100) if len(lang.wordList[i])> 0 ])
    words.extend([lang.wordList[i] for i in range(100) if len(lang.wordList[i])> 0 ])
    concepts.extend([i for i in range(100) if len(lang.wordList[i])> 0 ])
    names_langs_used.extend([lang.info["name"] for i in range(100) if len(lang.wordList[i])> 0 ])
    
#         #if lang.info["name"] in lang_names:
#     embeddings.extend([lang.info["embeddings"][i] for i in range(100) if len(lang.wordList[i])> 0 and i == concept_to_plot])
#     words.extend([lang.wordList[i] for i in range(100) if len(lang.wordList[i])> 0 and i == concept_to_plot])
#     concepts.extend([i for i in range(100) if len(lang.wordList[i])> 0 and  i == concept_to_plot])
#     names_langs_used.extend([lang.info["name"] for i in range(100) if len(lang.wordList[i])> 0 and i == concept_to_plot])
embeddings = np.array(embeddings).reshape(len(embeddings),-1)
print(embeddings.shape)
# print("calculating distances ....")
# from scipy.spatial.distance import cosine
# distances = np.zeros((len(embeddings),len(embeddings)))
# for a in range(distances.shape[0]):
#     print(a,distances.shape[0])
#     for e in range(distances.shape[0]):
#         distances[a,e] = 1-cosine(embeddings[a],embeddings[e])
# print("tsne ...")
# tsne_model = TSNE(2,1337,metric="precomputed")
# embeddings_tsne = tsne_model.fit_transform(distances)
print("tsne ...")
tsne_model = TSNE(2,1337,metric="euclidean")
embeddings_tsne = tsne_model.fit_transform(embeddings)
print("annotating plot ...")
for i in range(len(embeddings)):
    if concepts[i] == concept_to_plot: 
        plt.annotate(words[i]+"_"+str(concepts[i])+"_"+names_langs_used[i],(embeddings_tsne[i,0],embeddings_tsne[i,1]),color=(concepts[i]/100,((concepts[i]+50)%100)/100,concepts[i]/100))

plt.show()
