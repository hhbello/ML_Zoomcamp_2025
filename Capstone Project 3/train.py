import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("Starting training process...")

# 1. Load data
df = pd.read_csv('data/steel_plates_faults_original_dataset.csv')
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# 2. Preprocess
df['target'] = np.argmax(df[target_cols].values, axis=1)
X = df.drop(columns=target_cols + ['id', 'target'])
y = df['target']

feature_names = X.columns.tolist()

print(f"Features: {feature_names}")
print(f"Target classes: {target_cols}")

# 3. Split (using full train for final model, but keep validation for check)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train best model (XGBoost with tuned parameters)
params = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 200,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)

# 5. Evaluate on validation
val_acc = model.score(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}")

# 6. Save model and metadata
model_file = 'xgb_tuned.pkl'
with open(model_file, 'wb') as f:
    pickle.dump((model, feature_names, target_cols), f)

print(f"Model and metadata saved to {model_file}")

