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


# @ignore_warnings(category=ConvergenceWarning)
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
    imputer = IterativeImputer(max_iter=250)
    imputer.fit(X)
    return imputer.transform(X), y


def main(filename):
    X, y = preprocessing(filename)
    print("finish preprocessing")
    outer_cv = KFold(n_splits=5)
    best_score = None
    params = None
    for train_xi, test_xi in outer_cv.split(X):
        X_train, X_test = X[train_xi, :], X[test_xi, :]
        y_train, y_test = y[train_xi], y[test_xi]

        inner_cv = KFold(n_splits=5)

        model = Pipeline([
            ('over sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
            ('classifier', RandomForestClassifier())
        ])
        # scoring='accuracy'
        gridCV = GridSearchCV(estimator=model, param_grid=options_random_forest, scoring='accuracy', cv=inner_cv, refit=True)

        result = gridCV.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        y_predict = best_model.predict(X_test)
        # evaluate the model
        score = f1_score(y_test, y_predict)
        # store the result
        print("Current model:")
        pprint(result.best_params_)
        print("Score:")
        print(score)
        if best_score is None:
            best_score = score
            params = result.best_params_
        elif score > best_score:
            best_score = score
            params = result.best_params_
    print("Final result:")
    pprint(params)
    print("score:")
    print(best_score)
    clf = RandomForestClassifier(n_estimators=params.get("classifier__n_estimators"),
                                 max_depth=params.get("classifier__max_depth"))
    results = {"Accuracy": [],
               "F1-score": [],
               "Sensitivity": [],
               "Specificity": [],
               "AUROC": []}
    for i in range(1, 11):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= i)
        model = Pipeline([
            ('over sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
            ('model', clf)
        ])
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results["Accuracy"].append(accuracy_score(y_test, prediction))
        results["F1-score"].append(f1_score(y_test, prediction))
        results["Sensitivity"].append(sensitivity_score(y_test, prediction))
        results["Specificity"].append(specificity_score(y_test, prediction))
        results["AUROC"].append(roc_auc_score(y_test, prediction))
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
    pprint(results)

if __name__ == '__main__':
    main("./dataset.csv")
