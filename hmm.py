import nltk
import numpy as np

corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
partsOfSpeech = []
words = []
for sentence in corpus:
    for word in sentence:
        partsOfSpeech.append(word[1])
        words.append(word[0])
setOfSpeech = list(set(partsOfSpeech))
setOfWords = list(set(words))

transitionMatrix = np.zeros((len(setOfSpeech), len(setOfSpeech)))
observationMatrix = np.zeros((len(setOfSpeech),len(setOfWords) + 1))
initialStateMatrix = np.zeros(len(setOfSpeech))
previousPos = '';
for sentence in corpus:
    sentenceStartIdx = setOfSpeech.index(sentence[0][1])
    initialStateMatrix[sentenceStartIdx] = initialStateMatrix[sentenceStartIdx] + 1
    for word in sentence:
        wordIdx = setOfWords.index(word[0])
        posIdx = setOfSpeech.index(word[1])
        observationMatrix[posIdx,wordIdx] = observationMatrix[posIdx,wordIdx] + 1
        if previousPos != '':
            transitionMatrix[previousPos,posIdx] = transitionMatrix[previousPos,posIdx] + 1
            previousPos = posIdx
        else:
            previousPos = posIdx
initialStateMatrix += 1
observationMatrix += 1
transitionMatrix += 1

observationMatrix = observationMatrix/observationMatrix.sum(axis=1, keepdims=True)
transitionMatrix = transitionMatrix/transitionMatrix.sum(axis=1, keepdims=True)
initialStateMatrix = initialStateMatrix/sum(initialStateMatrix)