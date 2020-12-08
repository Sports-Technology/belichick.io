import pandas as pd
from sklearn.model_selection import train_test_split


def scrape_data(num_years):
    dataset_sz = num_years

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
            imput_list.append(col)

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

    y = df["Yards"]
    X = df.drop("Yards", axis=1)

    # split data to train and test
    x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.1)

    return dgci, x_training_data, x_test_data, y_training_data, y_test_data
