import numpy as np
import pandas as pd
from xgboost import predict_yards
from scrape import scrape_data

dgci, x_training_data, x_test_data, y_training_data, y_test_data = scrape_data(1)

'''

Cases:

1st - maximize mid-range yards
2nd and 3rd - choose option that is closest to guaranteeing 1st down
4th - field position, score, quarter

'''


def create_suggestion(situation):
    throw, rush = predict_yards(situation)

    down = situation["Down"].item()
    to_go = situation["ToGo"].item()
    yard_line = situation["YardLine"].item()

    if down < 3:
        if throw > rush:
            print("We recommend passing." + " Predicted Yards:", throw)
        else:
            print("We recommend rushing." + " Predicted Yards:", rush)
    elif down == 3:
        if rush >= to_go:
            print("We recommend rushing." + " Predicted Yards:", rush)
        else:
            print("We recommend passing." + " Predicted Yards:", throw)
    else:
        if yard_line < 10:
            if throw > to_go + 3:
                print("We recommend passing." + " Predicted Yards:", throw)
            elif rush > to_go + 3:
                print("We recommend rushing." + " Predicted Yards:", rush)
            else:
                print("We recommend punting")
        if rush > to_go:
            print("We recommend rushing." + " Predicted Yards:", rush)
        elif throw > to_go:
            print("We recommend passing." + " Predicted Yards:", throw)
        else:
            print("We recommend punting")


# example call
create_suggestion(dgci.iloc[20])
