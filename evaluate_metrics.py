from math import sqrt
from sklearn import metrics
import numpy as np
from scipy import stats

def mae(y, f):
    mae = metrics.mean_absolute_error(y, f)
    return mae

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def pearson(y, f):
    rp = stats.pearsonr(y, f)[0]
    return rp

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def r_squared(y, f):
    sse = np.sum((y - f) ** 2)
    ssr = np.sum((f - np.mean(y)) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst
    # r2 = metrics.r2_score(y, f)
    return r2

# 一致性指标
def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


# 分类模型评价指标
def accuracy_score(y, f):
    return metrics.accuracy_score(y, f)

def precision_score(y, f):
    return metrics.precision_score(y, f)

def recall_score(y, f):
    return metrics.recall_score(y, f)

def f1_score(y, f):
    return metrics.f1_score(y, f)

def mcc_score(y, f):
    return metrics.matthews_corrcoef(y, f)

def auc_score(y, f):
    roc_auc = metrics.roc_auc_score(y, f)
    return roc_auc

def auprc_score(y, f):
    p, r, _ = metrics.precision_recall_curve(y, f)
    auprc = metrics.auc(r, p)
    return auprc