from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from scrape import scrape_data
import pandas as pd
import numpy as np

dgci, x_training_data, x_test_data, y_training_data, y_test_data = scrape_data(1)

# xg boost regressor
xgboost = GradientBoostingRegressor()
xgboost.fit(x_training_data, y_training_data)
xg_y_hat = xgboost.predict(x_test_data)
xg_r2 = r2_score(y_test_data, xg_y_hat)
print(xg_r2)


def predict_yards(situation):
    # transpose for some reason
    situation = pd.DataFrame(situation).T

    cats = ['DefenseTeam', 'OffenseTeam', 'Formation', 'PlayType', 'PassType', 'YardLineDirection', "Quarter", "Down"]

    # The current options being considered to maximize
    PT = ["PASS", "RUSH"]

    preds = []

    prediction_values = pd.DataFrame(np.zeros([117, 117]), columns=np.asarray(x_test_data.columns))

    p_vals = pd.DataFrame(prediction_values.iloc[2]).T

    situation = pd.get_dummies(situation, columns=cats).drop("Yards", axis=1)

    for x in situation.columns:
        p_vals[x] = int(situation[x])

    for x in PT:
        dummy = p_vals.copy()
        if x == "PASS":
            dummy["IsPass"] = 1
            dummy["IsRush"] = 0
        else:
            dummy["IsPass"] = 0
            dummy["IsRush"] = 1
        dummy["PlayType_" + x] = 1
        preds.append(xgboost.predict(dummy)[0])

    return preds
