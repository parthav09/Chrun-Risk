import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
import itertools

from skopt import space
from functools import partial
from skopt import gp_minimize
import catboost as cb
import xgboost as xgb
import calc_metric
import lightgbm as lgb

import config

def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    # model = ensemble.RandomForestRegressor(**params)
    # model = cb.CatBoostRegressor(**params, verbose=0)
    model = lgb.LGBMClassifier(**params)

    scores = []

    for fold in range(5):

        # all columns are features except income and kfold columns
        features = [
            f for f in df.columns if f not in ("customer_id", "churn_risk_score", "kfold")
        ]

        # get training data using folds
        df_train = df[df.kfold != fold].reset_index(drop=True)

        # get validation data using folds
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # get training data
        x_train = df_train[features].values

        # get validation data
        x_valid = df_valid[features].values

        # fit model on training data
        model.fit(x_train, df_train.churn_risk_score)

        # predict on validation data
        valid_preds = model.predict(x_valid)

        # valid_preds = [int(i) for i in valid_preds]

        # calculate metric
        score = calc_metric.calc_score(df_valid.churn_risk_score.values, valid_preds)
        
        # append rmse in list
        scores.append(score)

    return -1 * np.mean(scores)

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)

    X = df.drop(["churn_risk_score"], axis=1).values
    y = df.churn_risk_score.values

    param_space = [
        space.Integer(5, 20, name="max_depth"),
        space.Real(0.01, 0.1, name="learning_rate"),
        space.Integer(100, 800, name="n_estimators"),
        space.Integer(70, 90, name="num_leaves")
    ]

    param_names = [
        "max_depth",
        "learning_rate",
        "n_estimators",
        "num_leaves"
    ]

    optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )

    print(dict(zip(param_names, result.x)))
