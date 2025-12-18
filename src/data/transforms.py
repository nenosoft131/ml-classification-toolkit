from src.data import preprocessing


class Normalize(object):
    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['spectrum'] = preprocessing.normalize(sample['spectrum'])
        else:
            sample = preprocessing.normalize(sample)
        return sample


class BaselineCorrect(object):
    def __init__(self,  lam=1e6, diff_order=2):
        self.lam = lam
        self.diff_order = diff_order

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['spectrum'] = preprocessing.baseline_correct(
                sample['spectrum'],
                lam=self.lam,
                diff_order=self.diff_order,
            )
        else:
            sample = preprocessing.baseline_correct(sample, lam=self.lam, diff_order=self.diff_order)
        return sample


class SmoothSavitzkiGoilay(object):
    def __init__(self, window=11, polyorder=2):
        self.window = window
        self.polyorder = polyorder

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['spectrum'] = preprocessing.smooth_savitzky_golay(
                sample['spectrum'],
                window=self.window,
                polyorder=self.polyorder
            )
        else:
            sample = preprocessing.smooth_savitzky_golay(
                sample,
                window=self.window,
                polyorder=self.polyorder
            )
        return sample


class DespikeWhitaker(object):
    def __init__(self, ma=10, threshold=50):
        self.ma = ma
        self.threshold = threshold

    def __call__(self, sample):
        if isinstance(sample, dict):
            sample['spectrum'] = preprocessing.despike_whitaker(
                sample['spectrum'],
                ma=self.ma,
                threshold=self.threshold
            )
        else:
            sample = preprocessing.despike_whitaker(
                sample,
                ma=self.ma,
                threshold=self.threshold
            )
        return sample


