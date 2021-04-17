import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from numpy import mean, std

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
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

options_random_forest = {
    "n_estimators": [*range(10, 40, 10)] + [*range(45, 105, 5)],
    "max_depth": [2 ** i for i in range(1, 7)]
}
options_xgboost = {
    "learning_rate": [0.01, 0.05, 0.1]
}


def split_to_data_and_target(df: pd.DataFrame):
    data = df.values
    X, y = data[:, 2:], data[:, 1]
    X = X.astype('float64')
    y = y.astype('int32')
    return X, y


def convert_sars_cov_exam_results_to_binary(df: pd.DataFrame):
    df.loc[df[df.columns[1]] == 'positive', [df.columns[1]]] = 1
    df.loc[df[df.columns[1]] == 'negative', [df.columns[1]]] = 0


def filter_nulls(df: pd.DataFrame):
    def predicate(x):
        for column in df.columns[2:]:
            if pd.notnull(x[column]):
                return True
        return False

    return df[df.apply(predicate, axis=1)]


def project_columns(df: pd.DataFrame):
    return df[df.columns[features]]


def impute(X):
    imputer = IterativeImputer(max_iter=250)
    imputer.fit(X)
    return imputer.transform(X)

def preprocessing(df):
    df = project_columns(df)
    df = filter_nulls(df)
    convert_sars_cov_exam_results_to_binary(df)
    X, y = split_to_data_and_target(df)
    return impute(X), y


def train(X, y):
    inner_cv = KFold(n_splits=5)
    outer_cv = KFold(n_splits=5)
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=options_random_forest, cv=inner_cv)
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='f1')
    print(nested_score)


def main(filename):
    df = pd.read_csv(filename)
    X, y = preprocessing(df)


if __name__ == '__main__':
    main("dataset.csv")
