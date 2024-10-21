import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

class XGBoostModel:
    def __init__(self, params=None):
        self.model = xgb.XGBClassifier(**params) if params else xgb.XGBClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        return {'accuracy': accuracy, 'f1_score': f1}