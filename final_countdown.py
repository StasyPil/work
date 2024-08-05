import dill
import pandas as pd
import sklearn.compose
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime

def filter_data(df):
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_time',
        'visit_date'
    ]
    return df.drop(columns_to_drop, axis = 1)

def main():
    df = pd.read_csv('data/hits_session.csv')

    X = df.drop(['target'], axis=1)
    y = df['target']

    numerical_features = sklearn.compose.make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = sklearn.compose.make_column_selector(dtype_include=['object'])

    numerical_transform = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transform, numerical_features),
        ('categorical', categorical_transform, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('column_transforer', column_transformer)
    ])

    models = [
        LogisticRegression(solver='saga'),
        RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt'),
        MLPClassifier(random_state=12, max_iter=500)
    ]

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])

        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score =score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    with open('model/client_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'final work prediction pipline',
                'autor': 'Anastasia Piletskaya',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score,
            },
        }, file)

if __name__ == '__main__':
    main()

