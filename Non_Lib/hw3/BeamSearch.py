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


def prune(BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore, PrunedPathScore = {}, {}
    scorelist = list(BlankPathScore.values()) + list(PathScore.values())
    scorelist = sorted(scorelist)
    cutoff = scorelist[-BeamWidth]
    for p in BlankPathScore.keys():
        if BlankPathScore[p] >= cutoff:
            PrunedBlankPathScore[p] = BlankPathScore[p]
    for p in PathScore.keys():
        if PathScore[p] >= cutoff:
            PrunedPathScore[p] = PathScore[p]
    return PrunedPathScore, PrunedBlankPathScore


def blank_extend(y, path_score, blank_path_score):
    blank_score_update = {}
    for p in blank_path_score.keys():
        blank_score_update[p] = (blank_path_score[p] * y[0])
    for p in path_score.keys():
        if p in blank_score_update.keys():
            blank_score_update[p] += path_score[p] * y[0]
        else:
            blank_score_update[p] = path_score[p] * y[0]
    return blank_score_update


def symbol_extend(y, path_score, blank_path_score, symbol_sets):
    path_update = set()
    score_update = {}

    for p in path_score.keys():
        for c in range(len(symbol_sets)):
            if symbol_sets[c] == p[-1]:
                new_path = p
            else:
                new_path = p + symbol_sets[c]
            score_update[new_path] = path_score[p] * y[c][0]
    for p in blank_path_score.keys():
        for c in range(len(symbol_sets)):
            new_path = p + symbol_sets[c]
            if new_path in score_update.keys():
                score_update[new_path] += blank_path_score[p] * y[c][0]
            else:
                score_update[new_path] = blank_path_score[p] * y[c][0]
    return score_update


def merge_identical(blank_path_score, path_score):
    MergedPath = path_score.keys()
    FinalPathScore = {}
    for p in MergedPath:
        FinalPathScore[p] = path_score[p]
    for p in blank_path_score.keys():
        if p in MergedPath:
            print("Found Identical")
            FinalPathScore[p] += blank_path_score[p]
        else:
            FinalPathScore[p] = blank_path_score[p]
    return FinalPathScore


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
    def path_init(Symbolsets, y_probs, BeamWitdth, PathScore, BlankPathScore):
        path = ""
        BlankPathScore[path] = y_probs[0][0][0]
        for c in range(len(SymbolSets)):
            path = SymbolSets[c]
            PathScore[path] = y_probs[c + 1][0][0]
        return prune(BlankPathScore, PathScore, BeamWidth)

    PathScore, BlankPathScore = path_init(SymbolSets, y_probs, BeamWidth, PathScore, BlankPathScore)
    # print(PathWithTerminalBlank, BlankPathScore)
    for i in range(1, seq_len):
        blank_score_update = blank_extend(y=y_probs[0, i], path_score=PathScore, blank_path_score=BlankPathScore)
        symbol_score_update = symbol_extend(y=y_probs[1:, i], path_score=PathScore, blank_path_score=BlankPathScore, symbol_sets=SymbolSets)
        PathScore, BlankPathScore = prune(blank_score_update, symbol_score_update, BeamWidth)
    FinalPathScore = merge_identical(PathScore, BlankPathScore)
    best_path, best_score = max(FinalPathScore.items(), key=lambda k: k[1])
    print(FinalPathScore)
    return best_path, FinalPathScore


if __name__ == "__main__":
    EPS = sys.float_info.epsilon
    y_rands = np.random.uniform(EPS, 1.0, (6, 20, 1))
    y_sum = np.sum(y_rands, axis=0)
    y_probs = y_rands / y_sum
    SymbolSets = ['a', 'b', 'c', 'd', 'e']
    # best_path, score = GreedySearch(SymbolSets, y_probs)
    BeamWidth = 3
    BestPath, BestScore = BeamSearch(SymbolSets, y_probs, BeamWidth)
