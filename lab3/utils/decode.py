import numpy as np
import importlib
def decode_segmap(pred,task):
    if task=='all':
        LABELS = importlib.import_module('utils.labels_all')
    elif task=='cat':
        LABELS = importlib.import_module('utils.labels_cat')
    elif task =='road':
        LABELS = importlib.import_module('utils.labels_road')
    pred = np.squeeze(pred)
    R = pred.copy()
    G = pred.copy()
    B = pred.copy()
    for _label in LABELS.labels:
        R[R==_label.trainId] = _label.outColor[0]
        G[G==_label.trainId] = _label.outColor[1]
        B[B==_label.trainId] = _label.outColor[2]
    img = np.zeros((pred.shape[0],pred.shape[1],3))
    img[:,:,0] = R/255
    img[:,:,1] = G/255
    img[:,:,2] = B/255

    return img


