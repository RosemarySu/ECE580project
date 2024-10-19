from sklearn.model_selection import train_test_split
from config import *
from utils import get_data, normalization

def load_data():
    X, y = get_data(TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, train_scaler = normalization(X_train, numerical_features)
    X_val = normalization(X_val, numerical_features, train_scaler)
    
    return X_train, y_train, X_val, y_val

def load_senti_data():
    X_senti, y_senti = get_data(SENTIMENT_INCLUDED_TRAIN_PATH)
    X_train_senti, X_val_senti, y_train_senti, y_val_senti = train_test_split(X_senti, y_senti, test_size=0.2, random_state=RANDOM_STATE)
    X_train_senti, train_scaler_senti = normalization(X_train_senti, numerical_features+senti_features)
    X_val_senti = normalization(X_val_senti, numerical_features+senti_features, train_scaler_senti)

    return X_train_senti, y_train_senti, X_val_senti, y_val_senti