from pprint import pprint

import pandas as pd
import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score
from imblearn.over_sampling import SVMSMOTE

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from ModelFactory import *

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
def find_best_hyperparams(X, y, model_factory):
    pipeline_name_classifier = 'classifier'
    pipeline_classifier_params_prefix = f'{pipeline_name_classifier}__'
    params_grid = convert_params_dict_to_pipeline_params(
        model_factory.get_params_grid(),
        pipeline_classifier_params_prefix
    )
    print(f"running {model_factory.name()}:")
    print('Params grid:')
    pprint(params_grid)

    outer_cv = KFold(n_splits=5)
    best_score = None
    best_params = None
    i = 0
    for train_xi, test_xi in outer_cv.split(X):
        X_train, X_test = X[train_xi, :], X[test_xi, :]
        y_train, y_test = y[train_xi], y[test_xi]

        inner_cv = KFold(n_splits=5)
        model = Pipeline([
            ('over_sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
            (pipeline_name_classifier, model_factory.create_default_classifier())
        ])
        gridCV = GridSearchCV(
            estimator=model,
            param_grid=params_grid,
            scoring='f1_macro',
            cv=inner_cv,
            refit=True
        )

        result = gridCV.fit(X_train, y_train)
        best_model = result.best_estimator_
        params = result.best_params_

        y_predict = best_model.predict(X_test)
        score = f1_score(y_test, y_predict, average='macro')

        print(f"inner cross validation iteration {i} params:")
        pprint(result.best_params_)
        print(f"score: {score}")

        if best_score is None or score > best_score:
            best_score = score
            best_params = params

        i += 1

    print("best params:")
    pprint(best_params)
    print(f"score: {best_score}")
    print('')

    return convert_pipeline_params_to_params_dict(best_params, pipeline_classifier_params_prefix)


def convert_params_dict_to_pipeline_params(params, prefix):
    result = {}
    for param in params.keys():
        result[f'{prefix}{param}'] = params[param]
    return result


def convert_pipeline_params_to_params_dict(params, prefix):
    result = {}
    param_name_start = len(prefix)
    for param in params.keys():
        result[param[param_name_start:]] = params[param]
    return result


def calculate_test_metrics(X, y, params, model_factory):
    results = {"Accuracy": [],
               "F1-score": [],
               "Sensitivity": [],
               "Specificity": [],
               "AUROC": []}
    retrain(X, y, params, model_factory, results)
    normalize_metric_results(results)
    return results


def retrain(X, y, params, model_factory, results):
    for i in range(1, 11):
        retrain_iter(X, y, params, model_factory, results, i)


def retrain_iter(X, y, params, model_factory, results, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = get_retrain_model(params, model_factory, i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    append_scores(results, y_test, prediction)


def get_retrain_model(params, model_factory, i):
    clf = model_factory.create_default_classifier()
    clf.set_params(**params)
    model = Pipeline([
        ('over sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
        ('model', clf)
    ])
    return model


def append_scores(results, y_test, prediction):
    def _f1_score(y_true, y_prediction):
        return f1_score(y_true, y_prediction, average='macro')
    score_evals = [
        ('Accuracy', accuracy_score),
        ('F1-score', _f1_score),
        ('Sensitivity', sensitivity_score),
        ('Specificity', specificity_score),
        ('AUROC', roc_auc_score),
    ]
    for score_eval in score_evals:
        results[score_eval[0]].append(score_eval[1](y_test, prediction))


def normalize_metric_results(results):
    for metric in results.keys():
        np_arr = np.array(results[metric])
        results[metric] = np.mean(np_arr), np.std(np_arr)


def print_all_results(results):
    models = [
        LogisticRegressionFactory,
        RandomForestFactory,
        XGBoostFactory,
        CatBoostFactory,
        LightGbmFactory,
    ]
    metrics = [
        'Accuracy',
        'F1-score',
        'Sensitivity',
        'Specificity',
        'AUROC'
    ]

    print('\nFinal results:\n')
    print('+-------------+-----------+-----------+-------------+-------------+-----------+')
    print_metric_headers(metrics)
    print('+-------------+-----------+-----------+-------------+-------------+-----------+')
    for model_factory_class in models:
        model = model_factory_class()
        if model.name() in results:
            print_model_results(metrics, model, results)
    print('+-------------+-----------+-----------+-------------+-------------+-----------+')


def print_metric_headers(metrics):
    print('| Model\\Score |', end='')
    for metric in metrics:
        print(f' {metric: <9} |', end='')
    print('')


def print_model_results(metrics, model, results):
    model_results = results[model.name()]
    print(f'| {model.name(): <11} |', end='')
    for metric in metrics:
        print_model_metric(metric, model_results)
    print('')


def print_model_metric(metric, model_results):
    value = model_results[metric]
    value_str = f'{value[0]:.2f}±{value[1]:.2f}'
    print(f' {value_str.ljust(max(9, len(metric)))} |', end='')


def train_models(X, y):
    models = [
        LogisticRegressionFactory,
        RandomForestFactory,
        XGBoostFactory,
        CatBoostFactory,
        LightGbmFactory,
    ]
    final_results = {}
    for model_factory_class in models:
        model_factory = model_factory_class()
        results = train_model(X, y, model_factory)
        final_results[model_factory.name()] = results
    return final_results


def train_model(X, y, model_factory):
    params = find_best_hyperparams(X, y, model_factory)
    results = calculate_test_metrics(X, y, params, model_factory)
    return results


def main(filename):
    df = pd.read_csv(filename)
    X, y = preprocessing(df)
    print("finish preprocessing")
    final_results = train_models(X, y)
    print_all_results(final_results)


if __name__ == '__main__':
    main("./dataset.csv")
