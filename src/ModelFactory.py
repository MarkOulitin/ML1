from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class ModelFactory:
    def name(self):
        raise NotImplementedError()

    def get_params_grid(self):
        raise NotImplementedError()

    def create_default_classifier(self):
        raise NotImplementedError()


class LogisticRegressionFactory(ModelFactory):
    def name(self):
        return "Logistic Regression"

    def get_params_grid(self):
        return {}

    def create_default_classifier(self):
        return LogisticRegression()


class RandomForestFactory(ModelFactory):
    def name(self):
        return "Random Forest"

    def get_params_grid(self):
        return {
            'n_estimators': list(range(10, 40, 10)) + list(range(45, 105, 5)),
            'max_depth': [2 ** i for i in range(1, 7)]
        }

    def create_default_classifier(self):
        return RandomForestClassifier()


class XGBoostFactory(ModelFactory):
    def name(self):
        return "XGBoost"

    def get_params_grid(self):
        return {
            "learning_rate": [0.01, 0.05, 0.1]
        }

    def create_default_classifier(self):
        return XGBClassifier()