import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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


def print_missing_values_percentage(df):
    # summarize the number of rows with missing values for each column
    for i in range(df.shape[1]):
        # count number of rows with missing values
        n_miss = df[[df.columns[i]]].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

def impute(X, y):
    # print total missing
    print('Missing: %d' % sum(np.isnan(X).flatten()))
    # define imputer
    imputer = IterativeImputer()
    # fit on the dataset
    imputer.fit(X)
    # transform the dataset
    Xtrans = imputer.transform(X)
    # print total missing
    print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))
    return Xtrans


def filter_by_features(feature_list, i, df):
    return df[df[feature_list[i]].notna()]


def changePosNegToNumber(y):
    return np.where(y == 'negative', np.float64(0), np.float64(0))


def main(filename):
    float_features = [
        'Hematocrit', 'Hemoglobin',
        'Platelets', 'Red blood Cells', 'Lymphocytes',
        'Mean corpuscular hemoglobin concentration (MCHC)',
        'Mean corpuscular hemoglobin (MCH)', 'Leukocytes', 'Basophils',
        'Eosinophils', 'Lactic Dehydrogenase', 'Mean corpuscular volume (MCV)',
        'Red blood cell distribution width (RDW)', 'Monocytes',
        'Mean platelet volume ', 'Neutrophils', 'Proteina C reativa mg/dL',
        'Creatinine', 'Urea', 'Potassium', 'Sodium', 'Aspartate transaminase',
        'Alanine transaminase',
    ]
    dtypes = {
        'Patient ID': np.dtype('U'),
        'SARS-Cov-2 exam result': np.dtype('U'),
    }
    for float_feature in float_features:
        dtypes[float_feature] = np.dtype('f')

    df = pd.read_csv(filename, dtype=dtypes)
    feature_list = [attribute for index, attribute in enumerate(list(df)) if index in features]
    df = df[df.columns[features]]
    df = filter_by_features(feature_list, 2, df)

    df.loc[df[df.columns[1]] == 'positive', [df.columns[1]]] = 1
    df.loc[df[df.columns[1]] == 'negative', [df.columns[1]]] = 0

    data = df.values
    X, y = data[:, 2:], data[:, 1]
    X = X.astype('float64')
    y = y.astype('int32')
    print(df.shape)
    impute(X, y)
    print(df.columns)
    # pipeline = Pipeline(steps=[('i', IterativeImputer()), ('m', RandomForestClassifier())])


if __name__ == '__main__':
    main("dataset.csv")