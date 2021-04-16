import pandas as pd
import numpy as np
from numpy import mean, std

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, KFold, GridSearchCV

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

options_random_forest ={"n_estimators": [*range(10, 40, 10)] + [*range(45, 105, 5)],
                        "max_depth": [2**i for i in range(1, 7)]}
options_xgboost = {"learning_rate": [0.01, 0.05, 0.1]}

def filter_by_features(feature_list, i, df):
    return df[df[feature_list[i]].notna()]


def changePosNegToNumber(y):
    return np.where(y == 'negative', 0, 1)

def preprocessing(filename):
    df = pd.read_csv(filename)
    feature_list = [attribute for index, attribute in enumerate(list(df)) if index in features]
    df = df[df.columns[features]]
    df = filter_by_features(feature_list, 2, df)
    data = df.values
    X, y = data[:, 2:], changePosNegToNumber(data[:, 1])
    imputer = IterativeImputer()
    imputer.fit(X)
    return imputer.transform(X), y


def main(filename):
    X, y = preprocessing(filename)
    # pipeline = Pipeline(steps=[('i', IterativeImputer()), ('m', RandomForestClassifier())])
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    print("Finish")
    # inner_cv = KFold(n_splits=5)
    # outer_cv = KFold(n_splits=5)
    # clf = GridSearchCV(estimator= RandomForestClassifier(), param_grid=options_random_forest, cv=inner_cv)
    # nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    # print(nested_score)


if __name__ == '__main__':
    main("./dataset.csv")
