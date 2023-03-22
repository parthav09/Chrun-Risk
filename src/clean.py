import pandas as pd
import numpy as np
import config
from sklearn import preprocessing

if __name__ == "__main__":

    train = pd.read_csv(config.TRAIN)
    test = pd.read_csv(config.TEST)

    train['churn_risk_score'] = train['churn_risk_score'].replace(-1, 1)

    # fill na
    dataset = [train, test]

    for data in dataset:

        # data.region_category = data.region_category.fillna("NONE")

        data['region_category'] = data['region_category'].fillna(pd.Series(np.random.choice(['Town', 'City', 'Village'],  p=[0.45, 0.40, 0.15], size=len(data))))

        # data.preferred_offer_types = data.preferred_offer_types.fillna("NONE")

        data['preferred_offer_types'] = data['preferred_offer_types'].fillna(pd.Series(np.random.choice(['Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers'],  p=[0.33, 0.33, 0.34], size=len(data))))

        data['joined_through_referral'] = data['joined_through_referral'].replace('?', np.nan)

        data['joined_through_referral'] = data['joined_through_referral'].fillna(pd.Series(np.random.choice(['Yes', 'No'],  p=[0.5, 0.5], size=len(data))))

        data.points_in_wallet = data.points_in_wallet.fillna(data.points_in_wallet.mean())

        data['avg_frequency_login_days'] = data['avg_frequency_login_days'].replace("Error", -1)

        data['avg_frequency_login_days'] = data['avg_frequency_login_days'].astype("float64")

        final_list = []
        for num in data['avg_frequency_login_days']:
            if num < 0 and num != -1:
                final_list.append(np.absolute(num))
            else:
                final_list.append(num)
        
        data['avg_frequency_login_days'] = final_list

        data['avg_time_login_day'] = data['avg_time_spent'] / data['avg_frequency_login_days']

        data['transaction_per_time'] = data['avg_transaction_value'] / data['avg_time_spent']

        categories = []
        for mem in data['membership_category']:
            categories.append(mem.split(' ')[0])
            
        data['membership_category'] = categories

        data = data.drop(['referral_id', 'joining_date', 'security_no'], axis=1)

        data['membership_spcl_discount'] = data['membership_category'].astype(str) + "_" + data['used_special_discount'].astype(str)

        data['offer_spcl_discount'] = data['offer_application_preference'].astype(str) + "_" + data['used_special_discount'].astype(str)

        data['offer_membership_type'] = data['membership_category'].astype(str) + "_" + data['offer_application_preference'].astype(str)

    test['churn_risk_score'] = -1

    combined_data = pd.concat([train, test])

    cat_cols = []
    for col in train.columns:
        if train[col].dtype == 'object' and col not in ['customer_id', 'Name']:
            cat_cols.append(col)

    for col in cat_cols:

        encoder = preprocessing.LabelEncoder()

        encoder.fit(combined_data[col])

        combined_data.loc[:, col] = encoder.transform(combined_data[col])

    train_cleaned = combined_data[combined_data.churn_risk_score != -1].reset_index(drop=True)
    test_cleaned = combined_data[combined_data.churn_risk_score == -1].reset_index(drop=True)

    test_cleaned = test_cleaned.drop(['churn_risk_score', 'Name'], axis=1)

    train_cleaned = train_cleaned.drop(['Name'], axis=1)

    train_cleaned['churn_risk_score'] = train_cleaned['churn_risk_score'].astype("int64")

    train_cleaned.to_csv("../input/train_cleaned.csv", index=False)

    test_cleaned.to_csv("../input/test_cleaned.csv", index=False)

