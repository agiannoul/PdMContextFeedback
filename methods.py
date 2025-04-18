
from sklearn.ensemble import IsolationForest as isolation_forest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from deepAnt.cnn_inner import CNN
from models.distance_based import distance_based_k_r as pb_k_r


def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst
def isolation_fores_semi(normal,target,*args, **kwargs):
    if 'random_state' in kwargs.keys():
        clf=isolation_forest(n_estimators=50,**kwargs)
    else:
        clf = isolation_forest(n_estimators=50, random_state=93)

    clf.fit(normal)
    return [-1*sc for sc in clf.score_samples(target).tolist()]


def ocsvm_semi(normal,target,*args, **kwargs):
    clf=OneClassSVM()
    clf.fit(normal)
    return [-1*sc for sc in clf.score_samples(target).tolist()]

def lof_semi(normal,target,*args, **kwargs):
    clf = LocalOutlierFactor(novelty=True,**kwargs)
    clf.fit(normal.values)
    return [-1 * sc for sc in clf.score_samples(target.values).tolist()]
def distance_based(normal,target,*args, **kwargs):
    clf=pb_k_r(k=1)
    clf.fit(normal.values)
    return clf.predict(target.values)


def deepAnt(normal,target,window_size=30,*args, **kwargs):
    clf = CNN(window_size=window_size, num_channel=[32, 32, 40], feats=normal.shape[1],
                   lr=0.0008, batch_size=128)
    clf.fit(normal.values.astype('float32'))
    score = clf.decision_function(target.values.astype('float32'))
    return score