import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pybaselines import Baseline


# The next function calculates the modified z-scores of a differentiated spectrum
def modified_z_score(ys):
    ysb = np.diff(ys)  # Differentiated intensity values
    median_y = np.median(ysb)  # Median of the intensity values
    median_absolute_deviation_y = np.median(
        [np.abs(y - median_y) for y in ysb])  # median_absolute_deviation of the differentiated intensity values
    if abs(median_absolute_deviation_y) < 0.001:
        return np.zeros_like(ys)
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ysb]  # median_absolute_deviationmodified z scores
    return modified_z_scores


def despike_whitaker(X: np.ndarray, ma=10, threshold=50):
    spikes = abs(np.array(modified_z_score(X))) > threshold
    spikes = np.hstack([spikes, np.zeros((len(spikes), 1))])
    X_out = X.copy()
    for k in np.arange(len(spikes)):
        for i in np.arange(spikes.shape[1]):
            if spikes[k, i]:
                w = np.arange(max(0, i - ma), min(spikes.shape[1], i + 1 + ma))
                we = w[spikes[k, w] == 0]
                X_out[k, i] = np.mean(X[k, we])
    return X_out


def smooth_savitzky_golay(y, window=11, polyorder=2):
    return savgol_filter(y, window_length=window, polyorder=polyorder, deriv=0)


def demean(X: np.ndarray):
    mean = X.mean(axis=1)
    print(mean.shape)
    X_zeroed = X - mean.reshape(-1, 1)
    return X_zeroed


def normalize(X: np.ndarray):
    X_out = X.copy()
    for k in range(X.shape[0]):
        norm = np.linalg.norm(X[k])
        if np.abs(norm) < 0.001 or norm != norm:
            norm = 1.0
        new = X[k] / norm
        # print(k, new.mean(), norm)
        X_out[k] = new * 100.0
    return X_out


def baseline_correct(X, lam=1e6, diff_order=2, return_baselines=False):
    baseline_fitter = Baseline(x_data=np.arange(X.shape[1]))
    baseline = np.zeros_like(X)
    for k in range(X.shape[0]):
        if np.abs(X[k].sum()) > lam:
            baseline[k] = baseline_fitter.arpls(X[k], diff_order=diff_order, lam=lam)[0]
    X_out = X - baseline
    if return_baselines:
        return X_out, baseline
    else:
        return X_out


if __name__ == '__main__':
    import data
    import config as cfg
    import matplotlib.pyplot as plt

    # data = pd.read_csv("~/Downloads/train_data.csv").iloc[100:110]
    #
    # data = data.iloc[:, 1:-1]
    #
    # data = smooth_savitzky_golay(data, window=20)
    # print(data)
    # plt.plot(data.values.T)
    # plt.show()

    dataset = data.load_data(
        data_root=cfg.data_root,
        groups=cfg.groups
    ).iloc[:10]

    data = dataset.iloc[:, :cfg.NUM_FREQUENCIES]
    # Y = data.iloc[:5, :cfg.NUM_FREQUENCIES].values
    # Y_new = despike(Y, ma=10)

    # data = despike_whitaker(data.iloc[:, :cfg.NUM_FREQUENCIES])
    # print(data.head())

    data = smooth_savitzky_golay(data, window=9)
    data = baseline_correct_ALS(data)
    data = normalize(data)

    plt.plot(data.values.T)
    plt.show()



    # Y_corrected = Y_new.copy()

    # estimated_baseline = baseline_als(Y_new, l, p)
    # plt.plot(estimated_baseline)
    # plt.show()
    exit()

    for k in range(len(Y_new)):
        # Estimation of the baseline:
        estimated_baseline = baseline_als(Y_new[k], l, p)

        # Baseline subtraction:
        baselined_spectrum = Y_new[k] - estimated_baseline

        smoothed_spectrum = smooth_sprectrum(baselined_spectrum)
        Y_corrected[k] = smoothed_spectrum

    # plt.plot(Y_new.T)
    plt.plot(Y_corrected.T)
    plt.show()
