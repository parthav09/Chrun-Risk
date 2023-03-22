import xgboost as xgb
from sklearn import ensemble
import catboost as cb
import lightgbm as lgb

models = {
    "xgboost": xgb.XGBClassifier(
        n_jobs=-1
        # max_depth=7,
        # n_estimators=600
    ),
    "rf": ensemble.RandomForestClassifier(
        n_estimators=600,
        max_depth=7
    ),
    "catboost": cb.CatBoostClassifier(
        verbose=0
        # max_depth=6
        # iterations=10000
    ),
    "lgbm": lgb.LGBMClassifier(
        n_jobs=-1
        # max_depth=7,
        # n_estimators=700,
        # num_leaves=80
        # learning_rate=0.062099410295319325
    )
}

# lgbm
# Current minimum: -73.7093
# {'max_depth': 13, 'learning_rate': 0.062099410295319325, 'n_estimators': 716, 'num_leaves': 83}