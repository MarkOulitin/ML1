import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from numpy import mean, std

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, KFold, GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
features = [0,
            2,
            6,
            7,
            8,
            10,
            11,
            12,
            15,
            13,
            14,
            16,
            92,
            17,
            19,
            18,
            9,
            39,
            41,
            42,
            40,
            43,
            44,
            48,
            47]

options_random_forest = {"n_estimators": [*range(10, 40, 10)] + [*range(45, 105, 5)],
                         "max_depth": [2 ** i for i in range(1, 7)]}
options_xgboost = {"learning_rate": [0.01, 0.05, 0.1]}


def filter_by_features(feature_list, i, df):
    return df[df[feature_list[i]].notna()]


def filter_nulls(df: pd.DataFrame):
    def predicate(x):
        for column in df.columns[2:]:
            if pd.notnull(x[column]):
                return True
        return False

    return df[df.apply(predicate, axis=1)]


def changePosNegToNumber(y):
    return np.where(y == 'negative', 0, 1)


@ignore_warnings(category=ConvergenceWarning)
def preprocessing(filename):
    df = pd.read_csv(filename)
    df = df[df.columns[features]]
    df = filter_nulls(df)

    df.loc[df[df.columns[1]] == 'positive', [df.columns[1]]] = 1
    df.loc[df[df.columns[1]] == 'negative', [df.columns[1]]] = 0

    data = df.values
    X, y = data[:, 2:], data[:, 1]
    X = X.astype('float64')
    y = y.astype('int32')
    imputer = IterativeImputer()
    imputer.fit(X)
    return imputer.transform(X), y


def main(filename):
    X, y = preprocessing(filename)
    # pipeline = Pipeline(steps=[('i', IterativeImputer()), ('m', RandomForestClassifier())])
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    inner_cv = KFold(n_splits=5)
    outer_cv = KFold(n_splits=5)
    clf = GridSearchCV(estimator= RandomForestClassifier(), param_grid=options_random_forest, cv=inner_cv)
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    print(nested_score)


if __name__ == '__main__':
    main("./dataset.csv")
