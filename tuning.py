# tuning.py
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import json
import os


def tune_xgboost(X_train, y_train, X_val, y_val, n_iter=100, cv=5):
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5)
    }


    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=n_iter, 
        scoring='f1', n_jobs=-1, cv=cv, verbose=1, random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X_train, y_train)

    val_score = best_model.score(X_val, y_val)

    return best_model, best_params, best_score, val_score


def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f)

def load_params(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def tune_and_save(X_train, y_train, X_val, y_val, params_file, n_iter=100, cv=5):
    best_model, best_params, best_score, val_score = tune_xgboost(X_train, y_train, X_val, y_val, n_iter, cv)
    save_params(best_params, params_file)
    return best_model, best_params, best_score, val_score