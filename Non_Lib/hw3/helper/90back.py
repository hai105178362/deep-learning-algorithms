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
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    PrunedPathWithTerminalBlank = set()
    PrunedPathWithTerminalSymbol = set()
    scorelist = []
    for p in BlankPathScore.keys():
        scorelist.append(BlankPathScore[p])
    for p in PathScore.keys():
        scorelist.append(PathScore[p])
    scorelist = sorted(scorelist)
    cutoff = scorelist[-BeamWidth]
    for p in BlankPathScore.keys():
        if BlankPathScore[p] >= cutoff:
            PrunedPathWithTerminalBlank.add(str(p))
            PrunedBlankPathScore[p] = BlankPathScore[p]
    for p in PathScore.keys():
        if PathScore[p] >= cutoff:
            PrunedPathWithTerminalSymbol.add(str(p))
            PrunedPathScore[p] = PathScore[p]
    # print(PrunedPathScore,PrunedBlankPathScore)
    # print(PrunedPathWithTerminalBlank,PrunedPathWithTerminalSymbol)
    # sys.exit(1)
    return PrunedPathWithTerminalBlank, PrunedPathWithTerminalSymbol, PrunedPathScore, PrunedBlankPathScore


def Blankextend(PathWithTerminalBlank, PathWithTerminalSymbol, y, PathScore, BlankPathScore, SymbolSets):
    blankpathupdate = set()
    blankscoreupdate = {}
    for p in BlankPathScore.keys():
        blankpathupdate.add(p)
        blankscoreupdate[p] = (BlankPathScore[p] * y[0])
    for p in PathScore.keys():
        if p in blankpathupdate or (blankpathupdate == set() and p == ""):
            blankscoreupdate[p] += PathScore[p] * y[0]
        else:
            blankpathupdate.add(p)
            blankscoreupdate[p] = PathScore[p] * y[0]
    return blankpathupdate, blankscoreupdate


def Symbolextend(PathWithTerminalBlank, PathWithTerminalSymbol, y, PathScore, BlankPathScore, symbolset):
    pathupdate = set()
    scoreupdate = {}
    newpath = ""

    for p in PathScore.keys():
        for c in range(len(symbolset)):
            if symbolset[c] == p[-1]:
                newpath = p
            else:
                newpath = p + symbolset[c]
            pathupdate.add(newpath)
            scoreupdate[newpath] = PathScore[p] * y[c][0]
    for p in BlankPathScore.keys():
        for c in range(len(symbolset)):
            newpath = p + symbolset[c]
            if newpath in pathupdate:
                scoreupdate[newpath] += BlankPathScore[p] * y[c][0]
            else:
                pathupdate.add(newpath)
                scoreupdate[newpath] = BlankPathScore[p] * y[c][0]
    return pathupdate, scoreupdate


def mergeIdentical(PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore):
    MergedPath = PathWithTerminalSymbol
    FinalPathScore = {}
    for p in MergedPath:
        FinalPathScore[p] = PathScore[p]
    for p in PathWithTerminalBlank:
        if p in MergedPath:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPath.add(p)
            FinalPathScore[p] = BlankPathScore[p]
    return MergedPath, FinalPathScore


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
        path = ""
        BlankPathScore[path] = y_probs[0][0][0]
        InitialPathsWithFinalBlank = set(path)
        InitialPathsWithFinalSymbol = set()
        for c in range(len(SymbolSets)):
            path = SymbolSets[c]
            PathScore[path] = y_probs[c + 1][0][0]
            InitialPathsWithFinalSymbol.add(path)
        return prune(InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, BlankPathScore, PathScore, BeamWidth)

    PathWithTerminalBlank, PathWithTerminalSymbol, PathScore, BlankPathScore = PathInit(SymbolSets, y_probs, BeamWidth, PathScore, BlankPathScore)
    # print(PathWithTerminalBlank, BlankPathScore)
    for i in range(1, seq_len):
        blankupdate, blankscoreupdate = Blankextend(PathWithTerminalBlank, PathWithTerminalSymbol, y=y_probs[0, i], PathScore=PathScore, BlankPathScore=BlankPathScore, SymbolSets=SymbolSets)
        symbolupdate, symbolscoreupdate = Symbolextend(PathWithTerminalBlank, PathWithTerminalSymbol, y=y_probs[1:, i], PathScore=PathScore, BlankPathScore=BlankPathScore, symbolset=SymbolSets)
        PathWithTerminalBlank, PathWithTerminalSymbol, PathScore, BlankPathScore = prune(blankupdate, symbolupdate, blankscoreupdate, symbolscoreupdate, BeamWidth)
    MergedPaths, FinalPathScore = mergeIdentical(PathWithTerminalBlank, PathWithTerminalSymbol, BlankPathScore, PathScore)
    BestPath, BestScore = max(FinalPathScore.items(), key=lambda k: k[1])
    print(FinalPathScore)
    return BestPath, FinalPathScore


if __name__ == "__main__":
    EPS = sys.float_info.epsilon
    y_rands = np.random.uniform(EPS, 1.0, (6, 20, 1))
    y_sum = np.sum(y_rands, axis=0)
    y_probs = y_rands / y_sum
    SymbolSets = ['a', 'b', 'c', 'd', 'e']
    # best_path, score = GreedySearch(SymbolSets, y_probs)
    BeamWidth = 3
    BestPath, BestScore = BeamSearch(SymbolSets, y_probs, BeamWidth)
