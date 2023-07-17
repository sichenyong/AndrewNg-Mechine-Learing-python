from sklearn import svm

def train(C, kernel = "linear"):
    mySvm = svm.SVC(C= C, kernel = kernel)
    return mySvm