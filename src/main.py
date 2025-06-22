import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess, model, utils, config

# Load
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test_X.csv')

# Clean
train = preprocess.clean_data(train)
test  = preprocess.clean_data(test, train_ref=train)

# Features
X = train[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
y = train[config.TARGET_COLUMN]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocess & Pipeline
preprocessor = preprocess.build_preprocessor()
pipeline = model.build_pipeline(preprocessor)
param_dist = model.get_param_dist(y)
search = model.tune_model(pipeline, X_train, y_train, param_dist)

# Eval
best_model = search.best_estimator_
utils.evaluate_model(best_model, X_val, y_val)

# Retrain & Predict
best_model.fit(X, y)
y_test = best_model.predict(test[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES])
utils.save_submission(test, y_test)
