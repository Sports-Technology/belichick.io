import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# import seaborn as sns
# import joblib
# import random
# import math


# Downloading Data
# The variable dataset size determines how big of a dataset is imported
# 1 = 2019 data, 2= 2019+2018, 3 = ...

dataset_sz = 1

stem = "../data/pbp-201"
suffix = ".csv"
sets = []


for x in range(dataset_sz):
    num = str(9 - x)
    name = stem + num + suffix
    sets.append(pd.read_csv(name))

data = pd.concat(sets)

len(data[data["Down"] == 4]["PlayType"])

# Used RapidMiner API to identify columns that correlate with target variable while
# also not having a large number of missing values (over 50% missing) or stability above 95%

good_columns = ['Minute', 'Second', 'OffenseTeam', 'DefenseTeam', 'Down', 'ToGo', 'YardLine',
                'SeriesFirstDown', 'Formation', 'PlayType', 'IsRush', 'IsPass',
                'PassType', 'YardLineFixed', 'YardLineDirection', 'Quarter', 'IsTouchdown', 'IsPenalty', 'Yards']

data_gc = data[good_columns]

# Create a standard time column  = minute*60+seconds
data_gc["Time"] = data_gc["Minute"] * 60 + data_gc["Second"]
data_gc.drop(["Minute", "Second"], axis=1, inplace=True)


# See which columns have NaN's in order to begin imputing missing values
imput_list = []
for col in data_gc.columns:
    x = data_gc[col].isna().value_counts()

    # Ignore column if no missing values
    if len(x) > 1:
        print(col)
        imput_list.append(col)
        print(x)


# Impute Pass Type columns with Misc
data_gc["PassType"].fillna('MISC', inplace=True)
# got tired of longer data frame name
dgci = data_gc


# Losing about 3000 out of 42,000 columns by dropping rows with NaN's

dgci = dgci.dropna()

# sanity check
for col in dgci.columns:
    x = dgci[col].isna().value_counts()

    # Ignore column if no missing values
    if len(x) > 1:
        print(col)
        print(x)

# If this prints something, something went wrong.
dgci["PlayType"].value_counts()

# Must turn categorical variables into dummy vars to make sklearn happy

# First see which columns are categorical (some already are made into dummy vars)
dgci["PlayType"].replace(to_replace="SACK", value="PASS", inplace=True)
dgci.info()

cats = ['DefenseTeam', 'OffenseTeam', 'Formation', 'PlayType', 'PassType', 'YardLineDirection', "Quarter", "Down"]
df = pd.get_dummies(dgci, columns=cats)

dgci.columns

y = df["Yards"]
X = df.drop("Yards", axis=1)
print(X.shape)

# split data to train and test
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.1)


# function that allows us to set up a prediction interval

def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile


# xg boost regressor
xgboost = GradientBoostingRegressor()
xgboost.fit(x_training_data, y_training_data)


# the column names that will be included in prediction
pred_cols = x_test_data.columns


# Input the information in the correct columns below to make a prediction, leave the play type column blank as the model will determine
# which play type maximizes the yards gained or points within a situation. Leave Yards blank too obv

# row of data used to predict
datum = dgci.iloc[5]

# transpose for some reason
datum = pd.DataFrame(datum).T


cats = ['DefenseTeam', 'OffenseTeam', 'Formation', 'PlayType', 'PassType', 'YardLineDirection', "Quarter", "Down"]

# The current options being considered to maximize
PT = ["PASS", "RUSH"]


preds = []

# not sure why
prediction_values = pd.DataFrame(np.zeros([117, 117]), columns=np.asarray(pred_cols))

p_vals = pd.DataFrame(prediction_values.iloc[5]).T

datum = pd.get_dummies(datum, columns=cats).drop("Yards", axis=1)

for x in datum.columns:
    p_vals[x] = int(datum[x])


print(p_vals.columns)

for x in PT:
    dummy = p_vals.copy()
    if x == "PASS":
        dummy["IsPass"] = 1
        dummy["IsRush"] = 0
    else:
        dummy["IsPass"] = 0
        dummy["IsRush"] = 1
    dummy["PlayType_" + x] = 1
    preds.append(xgboost.predict(dummy))

print(preds)

to_go = p_vals["ToGo"].iloc[0].item()

'''
* doesn't handle running down clock

Cases:

1st - maximize mid-range yards
2nd and 3rd - choose option that is closest to guaranteeing 1st down
4th - field position, score, quarter

'''

# first down
if p_vals["Down_1"].iloc[0].item() == 1:
    if pass_yards_mid > rush_yards_mid:
        print("We recommend passing." + " Predicted Yards:", pass_yards_mid)
    else:
        print("We recommend rushing." + " Predicted Yards:", rush_yards_mid)

# second or third down
elif p_vals["Down_2"].iloc[0].item() == 1 or p_vals["Down_3"].iloc[0].item() == 1:
    if rush_yards_lo > to_go:
        print("We recommend rushing." + " High Probability Predicted Yards:", rush_yards_lo)
    elif pass_yards_lo > to_go:
        print("We recommend passing." + " High Probability Predicted Yards:", pass_yards_lo)
    elif pass_yards_mid > rush_yards_mid:
        print("We recommend passing." + " Predicted Yards:", pass_yards_mid)
    else:
        print("We recommend rushing." + " Predicted Yards:", rush_yards_mid)

# fourth down, haven't written yet
