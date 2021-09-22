import numpy as np

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