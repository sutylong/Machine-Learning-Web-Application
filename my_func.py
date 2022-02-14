import numpy as np
import matplotlib.pyplot as plt

def Rsquared(y_true, y_predict):
    mean = np.sum(y_true)/len(y_true)
    ss_fit = np.sum((y_true - y_predict)**2)
    ss_mean = np.sum((y_true - mean)**2)
    R_squared = 1 - ss_fit/ss_mean

    return R_squared
