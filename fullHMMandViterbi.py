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

def viterbi(input, transitionMatrix, observationMatrix, initialStateMatrix):
    transitionMatrix = np.log(transitionMatrix)
    observationMatrix = np.log(observationMatrix)
    initialStateMatrix = np.log(initialStateMatrix)

    numStates = transitionMatrix.shape[0]
    T1 = np.empty((numStates, len(input)))
    T2 = np.empty((numStates, len(input)))

    for x in range(0, numStates):
        T1[x,0] = initialStateMatrix[x] * observationMatrix[x, input[0]]
        T2[x, 0] = 0

    for x in range (1, len(input)):
        for y in range(0, numStates):
            T1[y,x] = np.max(transitionMatrix[:, y] + T1[:, x-1] + observationMatrix[y, input[x]])
            T2[y, x-1] = np.argmax(np.exp2(transitionMatrix[:, y] + T1[:, x-1]))
    
    T1 = np.exp2(T1)
    bestRoute = np.zeros(len(input), dtype= 'int')
    bestRoute[-1] = np.argmax(T1[:,-1])
    for x in reversed(range(1, len(input))):
        bestRoute[x] = T2[bestRoute[x],x]
    return bestRoute

wordIdx = []

words = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
expected = []
for sentence in words:
    for word in sentence:
        if(word[0] in setOfWords):
            wordIdx.append(setOfWords.index(word[0]))
        else:
            wordIdx.append(len(setOfWords))
        expected.append(setOfSpeech.index(word[1]))

result = viterbi(wordIdx, transitionMatrix, observationMatrix, initialStateMatrix)
incorrect = 0
for idx in range(0, len(expected)):
    if expected[idx] != result[idx]:
        incorrect += 1
print((len(expected) - incorrect) / len(expected))