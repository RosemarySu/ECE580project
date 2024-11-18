import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
from config import *

def type_conversion(dataframe, categorical_features, numerical_features):
    dataframe[categorical_features] = dataframe[categorical_features].astype('category')
    dataframe[numerical_features] = dataframe[numerical_features].astype('float')
    return dataframe

def normalization(dataframe, numerical_features, scaler = None):
    if scaler:
        dataframe[numerical_features] = scaler.transform(dataframe[numerical_features])
        return dataframe
    scaler = StandardScaler()
    dataframe[numerical_features] = scaler.fit_transform(dataframe[numerical_features])
    return dataframe, scaler

def get_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    data = data[data['Quantity'] <=1]
    columns_to_drop = id_features + textual_features
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(existing_columns, axis=1)
    data = type_conversion(data, categorical_features, numerical_features)
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y

def clean_text(text):
    text = str(text)
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    return text

def clean_name(name):
    name = str(name)
    no_names = ["No Name Yet", "Nameless", "no_Name_Yet", "No Name Yet God Bless", "-no Name-", "[No Name]",
                "(No Name)", "No Names", "Not Yet Named"]
    for n in no_names:
        name.replace(n, "No Name")
    return name

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(['\s*'.join(key) 
                                               for key in mispell_dict.keys()]))
    return mispell_dict, mispell_re

def replace_typical_misspell(text, mispellings):
    def replace(match):
        return mispellings[re.sub('\s', '', match.group(0))]
    return mispellings.sub(replace, text)

def process_text_rnn(text):
    if text is None:
            return ''
    text = clean_text(text)
    text = replace_typical_misspell(text)
    for char in '()*,./:;\\\t\n':
        text = text.replace(char, '')
    text = re.sub('\s+', ' ', text)
    return text