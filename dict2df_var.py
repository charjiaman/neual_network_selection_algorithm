'''
function name: dict2df_var.py

dict2df_var.py generate data and export to csv so the training can use a fixed set of data for innitial stage of training.
refer to dic2df.py
'''

import pandas as pd

def dict2df(games_dic):
    #create empty matrix
    game_array = []
    #loop through agents and resources, and append values to matrix
    for game in games_dic['games']:
        game_data = {}
        for agent in game['agents']:
            game_data[agent] = 1

        for key, value in game['resources'].items(): #items contains key-value pair
            game_data[key] = value

        for key, value in game["action_set"].items():
            for res in value :
                game_data["action." + key + "." + res] = 1

       #loop through optimal_allocations and append each optimal allocation into separate rows
        for optimal in game["optimal_allocations"]:
            optimal_data = game_data.copy()  # create a copy of the game_data dictionary
            for key, value in optimal.items():
                if value is not None:
                    optimal_data["optimal." + key + "." + value] = 1
            game_array.append(optimal_data)  #append optimal_data as a new row

    df = pd.json_normalize(game_array)#convert into tabular format
    df = df.reindex(sorted(df.columns), axis = 1)#re-order so that action.x.x column is before optimal.x.x
    df = df.fillna(0)


    return df
                