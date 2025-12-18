from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, KFold, StratifiedKFold
import constants as cfg


def get_model(method, params=None, random_state=0):
    if params is None:
        params = {}

    if method == cfg.METHOD_SVM:
        # clf = svm.SVC(kernel='linear', C=100, random_state=42)
        c = float(params.get('C', 1000.0))
        clf = svm.SVC(C=c, random_state=random_state)
    elif method == cfg.METHOD_RANDOM_FOREST:
        n_estimators = params.get('num_trees', 100)
        max_depth = params.get('max_depth', None)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state)
    elif method == cfg.METHOD_LOGISTIC_REGRESSION:
        clf = LogisticRegression(random_state=random_state, penalty=params.get('penalty', 'l2'))
    elif method == cfg.METHOD_PERCEPTRON:
        clf = SGDClassifier(loss='perceptron', alpha=params.get('alpha', 0.1), learning_rate='adaptive', eta0=0.001, max_iter=500, random_state=random_state)
    elif method == cfg.METHOD_BOOSTING:
        print('Creating AdaBoostClassifier...')
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=params.get('boosting_depth', 2)), learning_rate=0.01, random_state=random_state)
    else:
        raise ValueError(f"Unknown classification method {method}")

    return clf


def classify_data(X, y, method, num_cv_splits=5, random_state=0, params=None):

    clf = get_model(method, params, random_state)

    # cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # cv = StratifiedKFold(n_splits=num_cv_splits, shuffle=True, random_state=random_state)

    # y_pred = cross_val_predict(clf, X, y, cv=cv)
    # print("Actual:\n", y.values)
    # print("Predicted:\n", y_pred)

    # scores = cross_val_score(clf, X, y, cv=cv)
    from st_util import run_cross_validation
    y_pred, scores = run_cross_validation(clf, X, y, num_cv_splits, random_state)
    print("\nScores:\n", scores)
    print(f"Accuracy: {scores.mean():.2f} (+-{scores.std():.2f})\n")

    return y_pred, {
        'cv_scores': scores,
        'acc': scores.mean(),
        'std': scores.std(),
    }


