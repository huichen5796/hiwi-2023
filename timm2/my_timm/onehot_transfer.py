import numpy as np

print('dododododoodododood')
def onehot_trans(label: float):
    if label == 1.0:
        return np.array([1.0, 0.0, 0.0, 0.0]).astype(int)
    elif label == 0.0:
        return np.array([0.0, 1.0, 0.0, 0.0]).astype(int)
    elif label == -1.0:
        return np.array([0.0, 0.0, 1.0, 0.0]).astype(int)
    else:
        return np.array([0.0, 0.0, 0.0, 1.0]).astype(int)