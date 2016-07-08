import numpy as np
import regex

def vectorLinspace(start,stop,num=50):
    assert len(start) == len(stop)
    assert num > 0
    
    return np.array([np.linspace(start[dim],stop[dim],num) for dim in range(len(start))]).transpose()


def getListofASJPPhonemes(word):
    phonemes_alone="pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieaouE3"
    phonemeSearchRegex = "["+phonemes_alone+"][\"\*]?(?!["+phonemes_alone+"]~|["+phonemes_alone+"]{2}\$)|["+phonemes_alone+"]{2}?~|["+phonemes_alone+"]{3}?\$"
    return regex.findall(phonemeSearchRegex, word)
