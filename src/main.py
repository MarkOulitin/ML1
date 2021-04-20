import time
from datetime import datetime
import sys
from pprint import pprint

import pandas as pd
import numpy as np
import shap
from imblearn.metrics import sensitivity_score, specificity_score
from imblearn.over_sampling import SVMSMOTE
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from ModelFactory import *

CV_INNER_N_ITERS = 5
CV_OUTER_N_ITERS = 5

features_short_names = [
    "HCT", "HGB", "PLT", "RBC", "LYM",
    "MCHC", "MCH", "WBC", "BAY", "EOS",
    "LDH", "MCV", "RWD", "MONO", "MPV",
    "NEU", "CRP", "CREAT", "Urea", "K+",
    "Na", "AST", "ALT"
]


def print_time_delta(t_s, t_e, lbl):
    if lbl != "":
        lbl = lbl + " "
    print(f'{lbl}time took: {seconds_to_string(t_e - t_s)}')


def print_current_time(lbl):
    if lbl != '':
        lbl = lbl + ': '
    print(f'{lbl}{datetime.now().strftime("%H:%M:%S")}')


def seconds_to_string(dt_s):
    if dt_s < 0:
        return f'<<negative time: {dt_s:.2f}>>'
    s = dt_s % 60
    mins = int(dt_s // 60)
    if mins == 0:
        return f'{s:.2f}s'
    hours = int(mins // 60)
    if hours == 0:
        return f'{mins}:{s:.2f} minutes'
    return f'{hours}:{mins}:{s:.2f} hours'



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


def project_columns(df: pd.DataFrame, features):
    return df[df.columns[features]]


# MAYBE def categorize_feature(df, feature_index, new_feature_name):
#     vals = df.iloc[:, feature_index]
#     mean = np.mean(vals)
#     std = np.std(vals)
#
#     df.loc[df[df.columns[feature_index]] < mean - 0.5*std, [new_feature_name]] = 0 # low
#     df.loc[df[df.columns[feature_index]] >= mean - 0.5*std, [new_feature_name]] = 1 # medium
#     df.loc[df[df.columns[feature_index]] > mean + 0.5*std, [new_feature_name]] = 2 # high
#     df[new_feature_name] = df[new_feature_name].astype('int32')
#     return mean, std


def impute(X_train, X_test, max_iter=500):
    imputer = IterativeImputer(max_iter=max_iter)
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test


def preprocessing(df):
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
    #  np.mean(np_arr), np.std(np_arr)
    # wbc_index = df.columns.get_loc("Leukocytes")
    # plt_index = df.columns.get_loc("Platelets")
    df = project_columns(df, features)
    df = filter_nulls(df)
    convert_exam_results_to_binary(df)
    featuresNames = [name for name in df.columns if not (name == 'Patient ID' or name == 'SARS-Cov-2 exam result')]
    X, y = split_to_data_and_target(df)
    return X, y
    # X = impute(X)
    # df = pd.DataFrame(data=X, columns=featuresNames)
    # wbc = df["Leukocytes"]
    # eos = df["Eosinophils"]
    # mono = df["Monocytes"]
    # lym = df["Lymphocytes"]
    # plt = df["Platelets"]
    # crp = df["Proteina C reativa mg/dL"]
    # rbc = df["Red blood Cells"]
    # hgb = df["Hemoglobin"]

    # print(f"WBC min = {np.min(wbc)}, WBC max = {np.max(wbc)}")
    # print(f"EOS min = {np.min(eos)}, EOS max = {np.max(eos)}")
    # print(f"MONO min = {np.min(mono)}, MONO max = {np.max(mono)}")
    # print(f"LYM min = {np.min(lym)}, LYM max = {np.max(lym)}")
    # print(f"PLT min = {np.min(plt)}, PLT max = {np.max(plt)}")
    # print(f"CRP min = {np.min(crp)}, CRP max = {np.max(crp)}")
    # print(f"rbc min = {np.min(rbc)}, RBC max = {np.max(rbc)}")
    # print(f"HGB min = {np.min(hgb)}, HGB max = {np.max(hgb)}")
    # logged_wbc = np.log([item + 100 for item in wbc])
    # wbc_mean = np.mean(logged_wbc)
    # wbc_std = np.std(logged_wbc)
    # logged_eos = np.log([item + 100 for item in eos])
    # eos_mean = np.mean(logged_eos)
    # eos_std = np.std(logged_eos)
    # logged_mono = np.log([item + 100 for item in mono])
    # mono_mean = np.mean(logged_mono)
    # mono_std = np.std(logged_mono)
    # logged_lym = np.log([item + 100 for item in lym])
    # lym_mean = np.mean(logged_lym)
    # lym_std = np.std(logged_lym)
    # logged_plt = np.log([item + 100 for item in plt])
    # plt_mean = np.mean(logged_plt)
    # plt_std = np.std(logged_plt)
    # logged_crp = np.log([item + 100 for item in crp])
    # crp_mean = np.mean(logged_crp)
    # crp_std = np.std(logged_crp)
    # logged_rbc = np.log([item + 100 for item in rbc])
    # rbc_mean = np.mean(logged_rbc)
    # rbc_std = np.std(logged_rbc)
    # logged_hgb = np.log([item + 100 for item in hgb])
    # hgb_mean = np.mean(logged_hgb)
    # hgb_std = np.std(logged_hgb)
    # 1. normalized = [(wbc_row * plt_row) / (wbc_mean * plt_mean) for wbc_row, plt_row in zip(logged_wbc, logged_plt)]
    # 2. normalized = [(plt_val + rbc_val) / (plt_mean * rbc_val) for plt_val, rbc_val in zip(logged_plt, logged_rbc)]
    # 3. normalized = [(wbc_val * eos_val * mono_val * lym_val) / (wbc_mean * eos_mean * mono_mean * lym_mean)  for wbc_val, eos_val, mono_val, lym_val in zip(logged_wbc, logged_eos, logged_mono, logged_lym)]
    # 4. normalized = [(wbc_val + eos_val + mono_val + lym_val) / (wbc_mean * eos_mean * mono_mean * lym_mean) for wbc_val, eos_val, mono_val, lym_val in zip(logged_wbc, logged_eos, logged_mono, logged_lym)]
    # normalized = [(plt_val - (rbc_val + wbc_val)) / (rbc_mean * plt_mean * wbc_mean) for crp_val, plt_val, wbc_val, rbc_val in zip(logged_crp, logged_plt, logged_wbc, logged_rbc)]
    # df['WBC*PLT / WBC.avg * PLT.avg'] = normalized
    # return df.values, y


# template code taken from https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
def find_best_hyperparams(X, y, model_factory):
    pipeline_name_classifier = 'classifier'
    pipeline_classifier_params_prefix = f'{pipeline_name_classifier}__'
    model_parms = model_factory.get_params_grid()
    params_grid = convert_params_dict_to_pipeline_params(
        model_parms,
        pipeline_classifier_params_prefix
    )
    print(f"starting to run {model_factory.name()}, ", end='')
    print_current_time('time')
    print('Params grid:')
    pprint(model_parms)

    time_outer_cv_start = time.perf_counter()

    outer_cv = KFold(n_splits=CV_OUTER_N_ITERS)
    best_score = None
    best_params = None
    i = 1
    for train_xi, test_xi in outer_cv.split(X):
        X_train, X_test = X[train_xi, :], X[test_xi, :]
        y_train, y_test = y[train_xi], y[test_xi]
        print(f"inner cross validation iteration {i}/{CV_INNER_N_ITERS} params of {model_factory.name()}:")
        time_iter_start = time.perf_counter()

        X_train, X_test = impute(X_train, X_test)
        inner_cv = KFold(n_splits=CV_INNER_N_ITERS)
        model = Pipeline([
            ('over_sampling', SVMSMOTE(sampling_strategy=1, k_neighbors=5)),
            (pipeline_name_classifier, model_factory.create_classifier())
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

        time_iter_end = time.perf_counter()
        pprint(result.best_params_)
        print(f"score: {score}")
        print_time_delta(time_iter_start, time_iter_end, 'iteration')

        if best_score is None or score > best_score:
            best_score = score
            best_params = params

        i += 1

    time_outer_cv_end = time.perf_counter()
    print("best params:")
    pprint(best_params)
    print(f"score: {best_score}")
    print_time_delta(time_outer_cv_start, time_outer_cv_end, f'model parameter tuning {model_factory.name()}')
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
        print(f'retraining iteration {i}/{10} of {model_factory.name()} model')
        retrain_iter(X, y, params, model_factory, results, i)


def retrain_iter(X, y, params, model_factory, results, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test = impute(X_train, X_test)
    model = get_retrain_model(params, model_factory, i)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    append_scores(results, y_test, prediction)


def get_retrain_model(params, model_factory, i):
    clf = model_factory.create_classifier()
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


def print_all_results(results, lbl, newline_after_label):
    models = [
        LogisticRegressionFactory,
        # RandomForestFactory,
        # XGBoostFactory,
        # CatBoostFactory,
        # LightGbmFactory,
    ]
    metrics = [
        'Accuracy',
        'F1-score',
        'Sensitivity',
        'Specificity',
        'AUROC'
    ]

    print(f'{lbl} results:')
    if newline_after_label:
        print('')
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


def shap_plot(model_factory, params):
    if not model_factory.should_plot_shap():
        print(f'model {model_factory.name()} not supporting SHAP')
        return

    print(f'generating SHAP for {model_factory.name()} model')
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test = impute(X_train, X_test)
        model = model_factory.create_classifier()
        model.set_params(**params)
        model = model.fit(X_train, y_train)
        shap_values = model_factory.shap_values(model, X_train, X_test)
        shap.summary_plot(
            shap_values,
            X_test,
            max_display=28,
            feature_names=features_short_names,
            show=False,
        )
        plt.savefig(f'shap-{model_factory.name()}.png')
    except Exception as e:
        print(f'Error while trying to plot SHAP of {model_factory.name()}')
        print(f'Error message: {e}')


def train_models(X, y):
    models = [
        LogisticRegressionFactory,
        RandomForestFactory,
        XGBoostFactory,
        # CatBoostFactory,
        LightGbmFactory,
    ]
    final_results = {}
    for model_factory_class in models:
        model_factory = model_factory_class()
        results = train_model(X, y, model_factory)
        final_results[model_factory.name()] = results
        print_all_results(final_results, 'Intermediate', newline_after_label=False)
        print('')
    return final_results


def train_model(X, y, model_factory):
    params = find_best_hyperparams(X, y, model_factory)
    results = calculate_test_metrics(X, y, params, model_factory)
    shap_plot(model_factory, params)
    return results


def start(filename):
    df = pd.read_csv(filename)
    time_preprocessing_start = time.perf_counter()
    X, y = preprocessing(df)
    time_preprocessing_end = time.perf_counter()
    print_time_delta(time_preprocessing_start, time_preprocessing_end, 'preprocessing')
    final_results = train_models(X, y)
    time_end = time.perf_counter()
    print('')
    print_all_results(final_results, 'Final', newline_after_label=True)
    return final_results, time_end


def main(filename):
    time_start = time.perf_counter()
    print_current_time('start time')
    final_results, time_end = start(filename)
    print_time_delta(time_start, time_end, 'total')
    print_current_time('end time')


if __name__ == '__main__':
    main("./dataset.csv")
    # df = pd.read_csv("./dataset.csv")
    # X, y = preprocessing(df)
    # factory = LightGbmFactory()
    #
    # # compute SHAP values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_test = impute(X_train, X_test)
    # model = factory.create_classifier()
    # model.set_params(n_estimators=4, max_depth=64, learning_rate=0.1)
    # model.fit(X_train, y_train)
    # shap_values = factory.shap_values(model, X_train, X_test)
    #
    # # explainer = shap.Explainer(
    # #     model.fit(X_train, y_train), X_train
    # # )
    # # shap_values = explainer(X_test)
    # # # explainer = shap.TreeExplainer(model.fit(X_train, y_train), X_train)
    # # # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(
    #     shap_values,
    #     X_test,
    #     max_display=28,
    #     feature_names=features_short_names,
    #     plot_size=(15, 15),
    #     show=False,
    # )
    # plt.savefig('shap-lgbm.png')
