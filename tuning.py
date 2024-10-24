from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.metrics import f1_score
from tqdm import tqdm
import json

class ProgressRandomSearchCV(RandomizedSearchCV):
    def _get_param_iterator(self):
        iterator = super()._get_param_iterator()
        return tqdm(iterator, desc="Hyperparameter search", total=self.n_iter)


def tune_xgboost(X_train, y_train, X_val, y_val, n_iter=100, cv=5):
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5)
    }

    xgb = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_jobs=-1)

    random_search = ProgressRandomSearchCV(
        xgb, param_distributions=param_dist, n_iter=n_iter, 
        scoring='f1_macro', n_jobs=-1, cv=cv, verbose=0, random_state=42
    )

    print(f"\nStarting hyperparameter search with {n_iter} iterations and {cv}-fold cross-validation...")
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("\nTraining final model with best parameters...")
    best_model = XGBClassifier(**best_params, eval_metric='mlogloss')
    best_model.fit(X_train, y_train)

    print("\nEvaluating on validation set...")
    val_pred = best_model.predict(X_val)
    val_score = f1_score(y_val, val_pred, average='macro')

    return best_model, best_params, best_score, val_score


def save_params(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f)

def load_params(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def tune_and_save(X_train, y_train, X_val, y_val, params_file, n_iter=100, cv=5):
    best_model, best_params, best_score, val_score = tune_xgboost(X_train, y_train, X_val, y_val, n_iter, cv)
    
    print("\nSaving best parameters...")
    save_params(best_params, params_file)
    
    print(f"\nBest cross-validation score: {best_score:.4f}")
    print(f"Validation score: {val_score:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    return best_model, best_params, best_score, val_score