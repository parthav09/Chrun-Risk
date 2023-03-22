from sklearn import metrics

def calc_score(actual, predicted):
    score = 100 * metrics.f1_score(actual, predicted, average="macro")
    return score