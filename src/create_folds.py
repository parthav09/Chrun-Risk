import pandas as pd
from sklearn import model_selection
import config

if __name__ == "__main__":

    df = pd.read_csv(config.TRAIN_CLEANED)

    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.churn_risk_score.values

    skf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv(config.TRAINING_FILE, index=False)