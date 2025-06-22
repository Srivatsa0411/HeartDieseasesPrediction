from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def build_pipeline(preprocessor):
    return Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])

def get_param_dist(y):
    imbalance_ratio = (y==0).sum() / (y==1).sum()
    return {
        'clf__n_estimators':        [500, 700, 900, 1100],
        'clf__max_depth':           [3, 4, 5, 6, 8],
        'clf__learning_rate':       [0.005, 0.01, 0.03, 0.05, 0.1],
        'clf__subsample':           [0.5, 0.6, 0.8, 1.0],
        'clf__colsample_bytree':    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__gamma':               [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
        'clf__reg_alpha':           [0, 0.1, 0.5, 1.0, 1.5, 2.0],
        'clf__reg_lambda':          [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'clf__min_child_weight':    [1, 3, 5, 7, 9, 11],
        'clf__max_delta_step':      [0, 1, 5, 7, 9, 10],
        'clf__scale_pos_weight':    [1, imbalance_ratio]
    }

def tune_model(pipeline, X_train, y_train, param_dist):
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=100,
        scoring='f1',
        cv=10,
        random_state=121,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    return search
