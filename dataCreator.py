import nba_api.stats
import nba_api
import nba_api.stats.static
import nba_api.stats.static.players
from nba_api.stats.endpoints import playercareerstats
import numpy as np
import pandas as pd
import matplotlib as plt

pd.set_option('display.max_columns', None)

playerdf = pd.DataFrame(nba_api.stats.static.players.get_active_players())
statdf = pd.DataFrame({'ptsAve' : [], 'tovAve' : [], 'blkAve' : [], 'stlAve' : [], 'astAve' : [], 'rebAve' : [], 'minAve' : []})

for index in playerdf.index:
    data = playercareerstats.PlayerCareerStats(playerdf.loc[index]['id']).get_data_frames()[1]
    row = []
    if data.empty:
        row = [-1, -1, -1, -1, -1, -1, -1]
    else:
        row.append(data.loc[0]['PTS'] / data.loc[0]['GP'])
        row.append(data.loc[0]['TOV'] / data.loc[0]['GP'])
        row.append(data.loc[0]['BLK'] / data.loc[0]['GP'])
        row.append(data.loc[0]['STL'] / data.loc[0]['GP'])
        row.append(data.loc[0]['AST'] / data.loc[0]['GP'])
        row.append(data.loc[0]['REB'] / data.loc[0]['GP'])
        row.append(data.loc[0]['MIN'] / data.loc[0]['GP'])
    statdf.loc[len(statdf)] = row

completedf = pd.merge(playerdf, statdf, left_index=True, right_index=True)
filtered_df = completedf[completedf['minAve'] >= 8.0]
print(len(filtered_df))

filtered_df.to_csv('/Users/mitch/Desktop/VSCode/NBAStatPredictor/DataSet.csv', index=False)