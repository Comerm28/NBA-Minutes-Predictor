import nba_api.stats.static.players
from nba_api.stats.endpoints import leaguedashplayerstats
import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)

playerdf = pd.DataFrame(nba_api.stats.static.players.get_active_players())

data = leaguedashplayerstats.LeagueDashPlayerStats(playerdf.loc[0]['id']).get_data_frames()[0]

print(data)