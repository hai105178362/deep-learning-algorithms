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
        if maxpos == 0:
            continue
        best_path += SymbolSets[maxpos - 1]
    return best_path, score


def prune(PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    print("Pruning")
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    PrunedPathWithTerminalBlank = ""
    PrunedPathWithTerminalSymbol = ""
    i = 0
    scorelist = []
    for p in PathWithTerminalBlank:
        scorelist.append(BlankPathScore[p])
    for p in PathWithTerminalSymbol:
        scorelist.append(PathScore[p])
    scorelist.sort()
    # print(scorelist)
    cutoff = scorelist[-BeamWidth]
    # print(cutoff)

    for p in PathWithTerminalBlank:
        if BlankPathScore[p] >= cutoff:
            PrunedPathWithTerminalBlank += str(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]
    for p in PathWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathWithTerminalSymbol += str(p)
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathWithTerminalBlank, PrunedPathWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


def Blankextend(PathWithTerminalBlank, PathWithTerminalSymbol, y, PathScore, BlankPathScore):
    blankpathupdate = set()
    blankscoreupdate = {}
    print("=============Extending==============")
    print(PathWithTerminalBlank, " | ", PathWithTerminalSymbol, " | ", y, " | ", PathScore, " | ", BlankPathScore)
    for p in PathWithTerminalBlank:
        blankpathupdate.add(p)
        blankscoreupdate[p] = (BlankPathScore[p] * y[0])
    for p in PathWithTerminalSymbol:
        if p in blankpathupdate:
            blankscoreupdate[p] += PathScore[p] * y[0]
        else:
            blankpathupdate.add(p)
            blankscoreupdate[p] = PathScore[p] * y[0]
    return blankpathupdate, blankscoreupdate


def Symbolextedn(PathWithTerminalBlank, PathWithTerminalSymbol, y, PathScore, BlankPathScore):
    # pathupdate = set()
    # scoreupdate = {}
    # for p in PathWithTerminalBlank:
    #     pathupdate.add(p)
    #     scoreupdate[p] = (BlankPathScore[p] * y[0])
    # for p in PathWithTerminalSymbol:
    #     if p in pathupdate:
    #         scoreupdate[p] += PathScore[p] * y[0]
    #     else:
    #         pathupdate.add(p)
    #         scoreupdate[p] = PathScore[p] * y[0]
    # return pathupdate, scoreupdate


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
    PathScore, BlankPathScore = {}, {}
    seq_len = len(y_probs[0])

    #################################################
    def PathInit(Symbolsets, y_probs, BeamWitdth, PathScore, BlankPathScore):
        path = " "
        # BlanckPathScore = {}
        # print(y_probs[:, 0])
        BlankPathScore[path] = y_probs[0][0][0]
        InitialPathsWithFinalBlank = set(path)
        InitialPathsWithFinalSymbol = set()
        # print(y_probs)
        # sys.exit(1)
        for c in range(len(SymbolSets)):
            path = SymbolSets[c]
            PathScore[path] = y_probs[c + 1][0][0]
            InitialPathsWithFinalSymbol.add(path)
        return prune(InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, BlankPathScore, PathScore, BeamWidth)

    #################################################
    PathWithTerminalBlank, PathWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore = PathInit(SymbolSets, y_probs, BeamWidth, PathScore, BlankPathScore)
    # print(PathWithTerminalBlank)
    # print(PathWithTerminalSymbol)
    # print(PrunedBlankPathScore)
    # print(PrunedPathScore)
    ##############################################
    print(y_probs[0])
    for i in range(1, seq_len):
        blankupdate, blankscoreupdate = Blankextend(PathWithTerminalBlank, PathWithTerminalSymbol, y_probs[0][i], PathScore, BlankPathScore)
        print(blankupdate, blankscoreupdate)
        sys.exit(1)

    return


if __name__ == "__main__":
    EPS = np.finfo(np.float).eps
    y_rands = np.random.uniform(EPS, 1.0, (4, 10, 1))
    y_sum = np.sum(y_rands, axis=0)
    y_probs = y_rands / y_sum
    SymbolSets = ['a', 'b', 'c']
    # best_path, score = GreedySearch(SymbolSets, y_probs)
    BeamWidth = 2
    BeamSearch(SymbolSets, y_probs, BeamWidth)
    # BestPath, MergedPathScores = BeamSearch(SymbolSets, y_probs, BeamWidth)
    # print(BestPath, MergedPathScores)
