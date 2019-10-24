import numpy as np
import sys


def GreedySearch(SymbolSets, y_probs):
    '''
    SymbolSets: This is the list containing all the symbols i.e. vocabulary (without blank)
    
    y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your
    batch size for part 1 would always remain 1, but if you plan to use
    your implementation for part 2 you need to incorporate batch_size.
    Return the forward probability of greedy path and corresponding compressed symbol
    sequence i.e. without blanks and repeated symbols.
    '''
    # print(y_probs)
    best_path = ""
    score = np.array([1.])
    seq_len = len(y_probs[0])
    # print(seq_len)
    # print(y_probs)
    for i in range(seq_len):
        maxpos = np.argmax(y_probs[:, i])
        maxprob = np.max(y_probs[:, i])
        score *= maxprob
        if maxpos==0:
            continue
        best_path += SymbolSets[maxpos-1]
        # score *= maxprob


    return best_path, score


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    '''
    SymbolSets: This is the list containing all the symbols i.e. vocabulary (without blank)
    
    y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your
    batch size for part 1 would always remain 1, but if you plan to use
    your implementation for part 2 you need to incorporate batch_size.
    BeamWidth: Width of the beam.
    The function should return the symbol sequence with the best path score (forward
    probability) and a dictionary of all the final merged paths with their scores.
    '''


if __name__ == "__main__":
    EPS = np.finfo(np.float).eps
    y_rands = np.random.uniform(EPS, 1.0, (4, 10, 1))
    y_sum = np.sum(y_rands, axis=0)
    y_probs = y_rands / y_sum
    SymbolSets = ['a', 'b', 'c']
    best_path, score = GreedySearch(SymbolSets, y_probs)
    print(best_path, score)
