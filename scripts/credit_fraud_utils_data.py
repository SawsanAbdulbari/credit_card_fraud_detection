import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_data(train_path, val_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    return train_data, val_data

def preprocess_data(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def resample_data(X, y, method='SMOTE', random_state=42):
    if method == 'RandomOverSampler':
        resampler = RandomOverSampler(random_state=random_state)
    elif method == 'RandomUnderSampler':
        resampler = RandomUnderSampler(random_state=random_state)
    elif method == 'SMOTE':
        resampler = SMOTE(random_state=random_state)
    else:
        return X, y
    X_res, y_res = resampler.fit_resample(X, y)
    return X_res, y_res
