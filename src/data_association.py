import numpy as np
from scipy.optimize import linear_sum_assignment

def assign_tracks(meas, preds):
    if len(preds) == 0 or len(meas) == 0:
        return []
    cost = np.linalg.norm(preds[:, None, :] - meas[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    return list(zip(row, col))
