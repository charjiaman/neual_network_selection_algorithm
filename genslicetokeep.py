'''
function: genslicetokeep.py

For each to_keep, generate games in dataframe format untill rows < 1200. Note, when iterates the game generater, for large to_keep, 40 iterations of games can result in 1000 rows of games entry in dataframe.
This code also stack the train, test and evaluation dataset so that the stacked train can be used to genereate the model and later the model will use to evaluate the test dataset for each game.
'''

import json
import pandas as pd
from datagen_var import SetCoverDataGenerator
from dict2df_var import dict2df
import random
from sklearn.model_selection import train_test_split
import numpy as np

random.seed(21)

def generate_game_data(to_keep):

    NUM_AGENTS = 6
    NUM_RESOURCES = 2 * NUM_AGENTS
    NUM_GAMES = 1
    games = SetCoverDataGenerator(NUM_RESOURCES, NUM_AGENTS, to_keep, NUM_GAMES)
    games_dic = json.loads(games)
    games_df = dict2df(games_dic)

    return games_df

def fill(data):
    data = data.reindex(sorted(data.columns), axis = 1) #re-order so that action.x.x column is before optimal.x.x
    data = data.fillna(0)
    data  = data.reset_index(drop=True)
    data = data.reindex(sorted(data.columns), axis=1)
    return data

def main():
     
    to_keep_list = []

    individual_dataframes = []
    individual_X_train = []
    individual_y_train = []
    individual_X_val = []
    individual_y_val = []
    individual_X_test = []
    individual_y_test = []

    #stacked data
    stacked_X_train = pd.DataFrame()
    stacked_X_val = pd.DataFrame()
    stacked_X_test = pd.DataFrame()
    stacked_y_train = pd.DataFrame()
    stacked_y_val = pd.DataFrame()
    stacked_y_test = pd.DataFrame()
   
    for to_keep in range(1, 73):
        print('to_keep', to_keep)
        
        each_game_df = pd.DataFrame()

        while len(each_game_df) < 50000:
            game_df = generate_game_data(to_keep)
            each_game_df = pd.concat([each_game_df, game_df], axis=0)

        each_game_df = fill(each_game_df)
        individual_dataframes.append(each_game_df)
        to_keep_list.append(to_keep)

    for i, df in enumerate(individual_dataframes):

        X = df.drop(columns=[col for col in df.columns if 'optimal' in col])  # feature data
        y = df[[col for col in df.columns if 'optimal' in col]]  # labels

        #split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        #further split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=21)

        individual_X_train.append(X_train)
        individual_y_train.append(y_train)
        individual_X_val.append(X_val)
        individual_y_val.append(y_val)
        individual_X_test.append(X_test)
        individual_y_test.append(y_test)

        #stack the dataframes for each category
        stacked_X_train = pd.concat([stacked_X_train, X_train], axis=0)
        stacked_X_val = pd.concat([stacked_X_val, X_val], axis=0)
        stacked_X_test = pd.concat([stacked_X_test, X_test], axis=0)
        stacked_y_train = pd.concat([stacked_y_train, y_train], axis=0)
        stacked_y_val = pd.concat([stacked_y_val, y_val], axis=0)
        stacked_y_test = pd.concat([stacked_y_test, y_test], axis=0)

        stacked_X_train = fill(stacked_X_train)
        stacked_X_val = fill(stacked_X_val)
        stacked_X_test = fill(stacked_X_test)      
        stacked_y_train = fill(stacked_y_train)
        stacked_y_val = fill(stacked_y_val)
        stacked_y_test = fill(stacked_y_test)

    dataframes = {
            'stacked_X_train': stacked_X_train,
            'stacked_X_val': stacked_X_val,
            'stacked_X_test': stacked_X_test,
            'stacked_y_train': stacked_y_train,
            'stacked_y_val': stacked_y_val,
            'stacked_y_test': stacked_y_test,
    }

    # Print the number of entries in each dataframe
    for name, df in dataframes.items():
            print(f"Number of entries in {name}: {len(df)}")

    #save the individual train, evaluate, and test dataframes to separate CSV files
    for i, X_train_df in enumerate(individual_X_train):
        X_train_df = X_train_df.reindex(columns=stacked_X_train.columns, fill_value=0)
        X_train_df.to_csv(f'individual_X_train_{i}.csv', index=False)
        individual_y_train[i] = individual_y_train[i].reindex(columns=stacked_y_train.columns, fill_value=0)
        individual_y_train[i].to_csv(f'individual_y_train_{i}.csv', index=False)
    for i, X_val_df in enumerate(individual_X_val):
        X_val_df = X_val_df.reindex(columns=stacked_X_val.columns, fill_value=0)
        X_val_df.to_csv(f'individual_X_val_{i}.csv', index=False)
        individual_y_val[i] = individual_y_val[i].reindex(columns=stacked_y_val.columns, fill_value=0)
        individual_y_val[i].to_csv(f'individual_y_val_{i}.csv', index=False)

    for i, X_test_df in enumerate(individual_X_test):
        X_test_df = X_test_df.reindex(columns=stacked_X_test.columns, fill_value=0)
        X_test_df.to_csv(f'individual_X_test_{i}.csv', index=False)
        individual_y_test[i] = individual_y_test[i].reindex(columns=stacked_y_test.columns, fill_value=0)
        individual_y_test[i].to_csv(f'individual_y_test_{i}.csv', index=False)

    #save the stacked train, valuate, and test dataframes to separate CSV files
    stacked_X_train.to_csv('stacked_X_train.csv', index=False)
    stacked_X_val.to_csv('stacked_X_val.csv', index=False)
    stacked_X_test.to_csv('stacked_X_test.csv', index=False)
    stacked_y_train.to_csv('stacked_y_train.csv', index=False)
    stacked_y_val.to_csv('stacked_y_val.csv', index=False)
    stacked_y_test.to_csv('stacked_y_test.csv', index=False)
    np.save('to_keep_list.npy', to_keep_list)

if __name__ == '__main__':
    main()
