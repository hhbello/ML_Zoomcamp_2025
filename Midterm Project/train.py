import pandas as pd
import numpy as np

import pickle

import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score



# Data prep

# load data from kaggle dataset
file_path = "concrete_data.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "zain280/concrete-data",
  file_path
)


# parameters
target = 'strength'
n_splits = 5


# Make columns lowercase and replace space
df.columns = df.columns.str.lower().str.replace(' ', '_')

# split dataset; train, val, and test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

X_full_train = df_full_train.drop('strength', axis=1)
y_full_train = df_full_train['strength']

X_train = df_train.drop(target, axis=1)
y_train = df_train[target]

X_val = df_val.drop(target, axis=1)
y_val = df_val[target]

X_test = df_test.drop(target, axis=1)
y_test = df_test[target]


# Training function
def train(X, y):
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=30
    )
    model.fit(X, y)

    return model

def predict(model, X_values):
    y_pred = model.predict(X_values)

    return y_pred


# KFold validation
print('KFold validation with n = {n_splits}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
rmse_scores = []
r2_scores = []
fold = 0


for train_idx, val_idx in kfold.split(df_full_train):
    df_train_fold = df_full_train.iloc[train_idx]
    df_val_fold = df_full_train.iloc[val_idx]

    X_train_fold = df_train_fold.drop('strength', axis=1)
    y_train_fold = df_train_fold['strength']

    X_val_fold = df_val_fold.drop('strength', axis=1)
    y_val_fold = df_val_fold['strength']

    model_fold = train(X_train_fold, y_train_fold)
    y_pred_fold = predict(model_fold, X_val_fold)

    rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    r2_fold = r2_score(y_val_fold, y_pred_fold)
    
    rmse_scores.append(rmse_fold)
    r2_scores.append(r2_fold)

    print(f'Fold {fold} - RMSE: {rmse_fold:.4f}, R2: {r2_fold:.4f}')
    fold = fold + 1

print('KFold Validation Results:')
print(f'Mean RMSE: {np.mean(rmse_scores):.4f} +- {np.std(rmse_scores):.4f}')
print(f'Mean R2: {np.mean(r2_scores):.4f} +- {np.std(r2_scores):.4f}')


# Training the model on the full training dataset
print('Training the model on the full training dataset')

et_model = train(X_full_train, y_full_train)
print('Model trained successfully!')

# Evaluate on the test set
print('Evaluating model on the test set')

y_pred_test = predict(et_model, X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f'Test Performance - RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}')


# Save the model
model_filename = 'et_model.bin'
with open(model_filename, 'wb') as f_out:
    pickle.dump(et_model, f_out)

print(f'The model is saved as {model_filename}')