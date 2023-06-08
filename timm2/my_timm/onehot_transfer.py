def onehot_trans(label: float):
    if label == 1.0:
        return [1.0, 0.0, 0.0, 0.0]
    elif label == 0.0:
        return [0.0, 1.0, 0.0, 0.0]
    elif label == -1.0:
        return [0.0, 0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 0.0, 1.0]
