import numpy as np 
import pandas as pd 

# Downloading Data
data = pd.read_csv("../data/pbp-2019.csv")

# Forming data structure purely from data we can extract
plays = dict()
count = 0
for i, j in data.iterrows():
    #if (data["Down"][i] == 0 or (data["IsRush"][i] == 0 and data["IsPass"][i] == 0)):
    #    continue

    plays[i] = {"GameID" : data["GameId"][i],
                "Time" : 
                [{
                    "Quarter" : data["Quarter"][i],
                    "Minute" : data["Minute"][i],
                    "Second" : data["Second"][i]
                }],
                "Down-and-Distance" : 
                [{
                    "Down" : data["Down"][i],
                    "ToGo" : data["ToGo"][i],
                    "YardLine" : data["YardLine"][i]
                }],
                "Decision" : 
                [{
                    "IsRush" : data["IsRush"][i],
                    "IsPass" : data["IsPass"][i]
                }],
                "Outcome" : 
                [{
                    "Yards" : data["Yards"][i],
                    "IsFirstDown" : data["SeriesFirstDown"][i]
                }]}

print(plays[0])