import pandas as pd
import config

lgbm4 = pd.read_csv("../submissions/lgbm4.csv")

lgbm = pd.read_csv("../submissions/lgbm.csv")

sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION)

sample_submission['churn_risk_score'] = [int(0.3*i + 0.7*j) for i, j in zip(lgbm['churn_risk_score'], lgbm4['churn_risk_score'])]

print(sample_submission.head(10))

sample_submission.to_csv("../submissions/0.3_lgbm_0.7_lgbm4.csv", index=False)