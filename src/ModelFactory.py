import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


class ModelFactory:
    def name(self):
        raise NotImplementedError()

    def get_params_grid(self):
        raise NotImplementedError()

    def create_classifier(self, **kwargs):
        raise NotImplementedError()

    def should_plot_shap(self):
        return True

    def __str__(self):
        return self.name()


class LogisticRegressionFactory(ModelFactory):
    def name(self):
        return 'LR'

    def get_params_grid(self):
        return {}

    def create_classifier(self, **kwargs):
        return LogisticRegression(**kwargs)

    def should_plot_shap(self):
        return False


class RandomForestFactory(ModelFactory):
    def name(self):
        return 'RF'

    def get_params_grid(self):
        return {
            'n_estimators': list(range(10, 40, 10)) + list(range(45, 105, 5)),
            'max_depth': [2 ** i for i in range(1, 7)],
        }

    def create_classifier(self, **kwargs):
        return RandomForestClassifier(**kwargs)


class XGBoostFactory(ModelFactory):
    def name(self):
        return 'XGBoost'

    def get_params_grid(self):
        return {
            'n_estimators': list(range(10, 40, 10)) + list(range(45, 105, 5)),
            'max_depth': [2 ** i for i in range(1, 7)],
            'learning_rate': [0.01, 0.05, 0.1]
        }

    def create_classifier(self, **kwargs):
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)


class CatBoostFactory(ModelFactory):
    def name(self):
        return 'CatBoost'

    def get_params_grid(self):
        return {
            'n_estimators': list(range(10, 40, 10)) + list(range(45, 105, 5)),
            'max_depth': [2, 4, 8],
            'learning_rate': [0.01, 0.05, 0.1]
        }

    def create_classifier(self, **kwargs):
        return CatBoostClassifier(verbose=False, allow_writing_files=False, **kwargs)


class LightGbmFactory(ModelFactory):
    def name(self):
        return 'LightGBM'

    def get_params_grid(self):
        return {
            'n_estimators': list(range(10, 40, 10)) + list(range(45, 105, 5)),
            'max_depth': [2 ** i for i in range(1, 7)],
            'learning_rate': [0.01, 0.05, 0.1]
        }

    def create_classifier(self, **kwargs):
        return LGBMClassifier(objective='binary', **kwargs)
