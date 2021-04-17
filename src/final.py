from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score
from imblearn.over_sampling import SMOTE, SVMSMOTE
from numpy import mean, std

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, KFold, GridSearchCV, train_test_split
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

options_random_forest = {'classifier__n_estimators': list(range(10, 40, 10)) + list(range(45, 105, 5)),
                         'classifier__max_depth': [2 ** i for i in range(1, 7)]}
options_xgboost = {"learning_rate": [0.01, 0.05, 0.1]}


def split_to_data_and_target(df: pd.DataFrame):
    data = df.values
    X, y = data[:, 2:], data[:, 1]
    X = X.astype('float64')
    y = y.astype('int32')
    return X, y


def convert_exam_results_to_binary(df: pd.DataFrame):
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
    convert_exam_results_to_binary(df)
    X, y = split_to_data_and_target(df)
    return impute(X), y


# template code taken from https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
def find_best_hyperparams(X, y):
    outer_cv = KFold(n_splits=5)
    best_score = None
    best_params = None
    for train_xi, test_xi in outer_cv.split(X):
        X_train, X_test = X[train_xi, :], X[test_xi, :]
        y_train, y_test = y[train_xi], y[test_xi]

        inner_cv = KFold(n_splits=5)
        model = Pipeline([
            ('over_sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
            ('classifier', RandomForestClassifier())
        ])
        gridCV = GridSearchCV(
            estimator=model,
            param_grid=options_random_forest,
            scoring='f1',
            cv=inner_cv,
            refit=True
        )

        result = gridCV.fit(X_train, y_train)
        best_model = result.best_estimator_
        params = result.best_params_

        y_predict = best_model.predict(X_test)
        score = f1_score(y_test, y_predict)

        print("Current model:")
        pprint(result.best_params_)
        print("Score:")
        print(score)

        if best_score is None or score > best_score:
            best_score = score
            best_params = params

    print("Final result:")
    pprint(best_params)
    print("score:")
    print(best_score)

    return best_params


def calculate_test_metrics(X, y, params):
    results = {"Accuracy": [],
               "F1-score": [],
               "Sensitivity": [],
               "Specificity": [],
               "AUROC": []}
    retrain(X, y, params, results)
    normalize_metric_results(results)
    return results


def retrain(X, y, params, results):
    for i in range(1, 11):
        retrain_iter(X, y, params, results, i)


def retrain_iter(X, y, params, results, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = get_retrain_model(params, i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    append_scores(results, y_test, prediction)


def get_retrain_model(params, i):
    clf = RandomForestClassifier(
        n_estimators=params.get("classifier__n_estimators"),
        max_depth=params.get("classifier__max_depth")
    )
    model = Pipeline([
        ('over sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
        ('model', clf)
    ])
    return model


def append_scores(results, y_test, prediction):
    results["Accuracy"].append(accuracy_score(y_test, prediction))
    results["F1-score"].append(f1_score(y_test, prediction))
    results["Sensitivity"].append(sensitivity_score(y_test, prediction))
    results["Specificity"].append(specificity_score(y_test, prediction))
    results["AUROC"].append(roc_auc_score(y_test, prediction))


def normalize_metric_results(results):
    temp = np.array(results["Accuracy"])
    results["Accuracy"] = np.mean(temp), np.std(temp)
    temp = np.array(results["F1-score"])
    results["F1-score"] = np.mean(temp), np.std(temp)
    temp = np.array(results["Sensitivity"])
    results["Sensitivity"] = np.mean(temp), np.std(temp)
    temp = np.array(results["Specificity"])
    results["Specificity"] = np.mean(temp), np.std(temp)
    temp = np.array(results["AUROC"])
    results["AUROC"] = np.mean(temp), np.std(temp)


def main(filename):
    X, y = preprocessing(filename)
    print("finish preprocessing")
    params = find_best_hyperparams(X, y)
    results = calculate_test_metrics(X, y, params)
    pprint(results)


if __name__ == '__main__':
    main("./dataset.csv")
