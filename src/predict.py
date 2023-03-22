import pandas as pd
import numpy as np
import itertools
import joblib
import os

from sklearn import preprocessing

import config


def get_preds(test):

    test_preds = 0

    model_path = os.path.join(config.MODEL_OUTPUT, "lgbm/")

    for fold in range(5):

        model = joblib.load(os.path.join(model_path, f"lgbm_{fold}.bin"))

        temp_preds = model.predict(test)

        test_preds += temp_preds / 5

    return test_preds

def main():

    # read test and sample_submission
    df_test = pd.read_csv(config.TEST_CLEANED)
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION)
    sample_submission['customer_id'] = df_test.customer_id


    # all columns are features except income and kfold columns
    features = [
        f for f in df_test.columns if f not in ("customer_id", "churn_risk_score")
    ]

    print(df_test.head())

    # get test
    test = df_test[features].values

    final_preds = get_preds(test)

    sample_submission['churn_risk_score'] = [int(i) for i in final_preds]

    print(sample_submission.head(10))

    sample_submission.to_csv("../submissions/lgbm7.csv", index=False)

if __name__ == "__main__":
    main()