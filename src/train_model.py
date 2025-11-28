import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
DATA_PATH = os.path.join('data','WA_Fn-UseC_-Telco-Customer-Churn.csv')

print('Loading data from', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Basic cleaning
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

X = df.drop(columns=['Churn'])
y = df['Churn']

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

print('Numerical columns:', num_cols)
print('Categorical columns:', cat_cols)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols),
                                               ('cat', cat_transformer, cat_cols)])

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))
])

param_grid = {
    'classifier__n_estimators':[100],
    'classifier__max_depth':[4],
    'classifier__learning_rate':[0.1]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

print('Starting training...')
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print('Best params:', grid.best_params_)
print('Best CV score:', grid.best_score_)

best_model = grid.best_estimator_
probs = best_model.predict_proba(X_test)[:,1]
print('Test ROC AUC:', roc_auc_score(y_test, probs))

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/churn_model.pkl')
print('Saved model to models/churn_model.pkl')