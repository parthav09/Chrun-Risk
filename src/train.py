import pandas as pd
import numpy as np

import model_dispatcher
from sklearn import metrics
from sklearn import preprocessing
import calc_metric

import argparse
import os

import joblib

import config

def run(fold, arg_model):

    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # all columns are features except id, target and kfold
    features = [
        f for f in df.columns if f not in ("customer_id", "churn_risk_score", "kfold")
    ]

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].valuesx2

    # get validation data
    x_valid = df_valid[features].values

    # y_valid
    y_valid = df_valid.churn_risk_score.values

    # initialize model
    model = model_dispatcher.models[arg_model]

    # fit model on training data (lbl)
    model.fit(x_train, df_train.churn_risk_score.values)

    # predict on validation data
    valid_preds = model.predict(x_valid)

    # get f1 score
    score = calc_metric.calc_score(y_valid, valid_preds)
    
    # print auc
    print(f"Fold = {fold}, F1 score = {score}")

    # save the model
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"lgbm/{arg_model}_{fold}.bin")
    )

    return score

if __name__ == "__main__":

    scores = []

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type

    parser.add_argument(
        "--model",
        type=str
    )

    # read the arguments from command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    for fold_ in range(5):
        score = run(fold=fold_, arg_model=args.model)
        scores.append(score)

    print(f"Average score: {np.mean(scores)}")
