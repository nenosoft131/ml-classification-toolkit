import streamlit_app as st
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, KFold, StratifiedKFold

import config as cfg
import preprocessing
from data import load_dataset


@st.cache
def cached_load_data(dataset):
    return load_dataset(dataset)


@st.cache
def compute_tsne(X):
    print(len(X))
    return TSNE(
        n_components=2,
        learning_rate='auto',
        random_state=0,
        perplexity=min(30.0, len(X)-1)
    ).fit_transform(X)

@st.cache
def run_cross_validation(clf, X, y, num_cv_splits, random_state=0):
    cv = StratifiedKFold(n_splits=num_cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    scores = cross_val_score(clf, X, y, cv=cv)
    return y_pred, scores

@st.cache
def run_filtering(X):
    X =  preprocessing.despike_whitaker(X)
    X =  preprocessing.smooth_savitzky_golay(X)

@st.cache
def run_baseline_correction(X):
    return preprocessing.baseline_correct(X)
