import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from config import EMAIL_DATASET

def preprocess_data():
    df = pd.read_csv(EMAIL_DATASET)
    df.loc[df.sample(frac=0.1).index, 'sender_importance'] = np.nan
    imputer = SimpleImputer()
    df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])
    X = df.drop(columns=['label'])
    y = df['label'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2)
