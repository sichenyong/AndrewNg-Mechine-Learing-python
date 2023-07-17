import numpy as np

def selectThreshold(yval, pval):
    yval = yval.flatten()
    pval = pval.flatten()
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval) + stepsize, stepsize):
        predictions = (pval < epsilon).astype(int)
        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1