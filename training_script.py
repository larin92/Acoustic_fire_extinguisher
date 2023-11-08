import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io.arff import loadarff 
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from joblib import dump
import json
import warnings

# Constants
RAND_SEED = 42
DATASET_PATH = 'dataset/Acoustic_Extinguisher_Fire_Dataset.arff'
TRAINING_PARAMS_PATH = 'best_xgb_params.json'
MODEL_DV_PATH = 'model_dv.pkl'

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(path) -> pd.DataFrame:
    raw_data = loadarff(path)
    df = pd.DataFrame(raw_data[0])
    df['FUEL'] = df['FUEL'].apply(lambda s: s.decode('utf-8') if isinstance(s, bytes) else s)
    df['CLASS'] = df['CLASS'].apply(lambda i: int(i) if isinstance(i, bytes) else i)
    return df

def load_params(path) -> dict:
    with open(path, 'r') as file:
        params = json.load(file)
    return params

def prepare_split(df, seed, train_frac, test_frac):
    random_generator = np.random.default_rng(seed=seed)
    sample = lambda frac: df.sample(frac=frac, 
                                    random_state=random_generator, 
                                    ignore_index=True)
    
    df_train, df_test = sample(train_frac), sample(test_frac)
    y_train = df_train.pop('CLASS')
    y_test = df_test.pop('CLASS')

    dv = DictVectorizer()
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))
    X_test = dv.transform(df_test.to_dict(orient='records'))
    
    return X_train, y_train, X_test, y_test, dv

def train_model(X_train, y_train, X_test, y_test, training_params):
    model = xgb.XGBClassifier(**training_params)
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    test_auc_score = roc_auc_score(y_test, y_test_pred)
    print(f"Test AUC Score from XGBoost model: {test_auc_score}")

    return model

def main():
    data = load_data(DATASET_PATH)
    training_params = load_params(TRAINING_PARAMS_PATH)
    
    X_train, y_train, X_test, y_test, dv = prepare_split(data, RAND_SEED, 0.8, 0.2)
    model = train_model(X_train, y_train, X_test, y_test, training_params)
    
    # Save the model and DictVectorizer to a file
    dump((model, dv), MODEL_DV_PATH)

if __name__ == '__main__':
    main()
