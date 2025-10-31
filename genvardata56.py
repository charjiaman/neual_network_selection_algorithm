'''
function name: genvardata56.py

generate different set of games eg different number of agents, resources and concate all games together with extra paddings if none.
refer to genfixdata.py

generate vardata with to keep upper limit to 56
'''

import json #data structure as key value pairs and array
import pandas as pd #import python data structure 

'''import user defined modules'''
from datagen_var import SetCoverDataGenerator #dataset generator from https://github.com/joshuahseaton/SubmodOptDatasetGen.git
from dict2df_var import dict2df #convert data generated from datagen.py

import random
random.seed(21)


def main():

    num_iterations = 10
    games_df = pd.DataFrame() 
    all_games_df = pd.DataFrame()
    to_keep_list = []
    
    for i in range(num_iterations):
        print('genvardata56 iteration', i)
        with open("output56.txt", "a") as file:
            # Write the iteration number 'i' to the file
            file.write(f"iteration {i}\n")
        #NUM_RESOURCES = 2*NUM_AGENTS, to_keep < NUM_AGENTS * NUM_RESOURCES
        #generate random numbers for NUM_RESOURCES, NUM_AGENTS, to_keep, and NUM_GAMES
        NUM_AGENTS = 2
        NUM_RESOURCES = 2*NUM_AGENTS
        to_keep = random.randint(2, 56)
        NUM_GAMES = 1
        '''obtain proper form of train test data'''
        #load json file which are all game examples with ground truth
        games = SetCoverDataGenerator(NUM_RESOURCES, NUM_AGENTS, to_keep, NUM_GAMES)
        #print(games)
        #convert to dictionary
        games_dic = json.loads(games)
        games_df = dict2df(games_dic)
        #print(f'Converted to DataFrame:\n{games_df}')
        to_keep_list.append(to_keep)
        # Append the DataFrame to games_df
        all_games_df = pd.concat([all_games_df, games_df], axis=0)
    #print(to_keep_list)

    all_games_df = all_games_df.reindex(sorted(all_games_df.columns), axis = 1) #re-order so that action.x.x column is before optimal.x.x
    all_games_df = all_games_df.fillna(0)
    all_games_df  = all_games_df.reset_index(drop=True)
    all_games_df = all_games_df.reindex(sorted(all_games_df.columns), axis=1)
    #print(all_games_df)
    all_games_df.to_csv('./'+'vardata56.csv') #output games_df to csv files
    to_keep_counts = pd.Series(to_keep_list).value_counts().reset_index()

    print(all_games_df.info())

    #count how many games have how many agents
    to_keep_counts = to_keep_counts.rename(columns={'index': 'to_keep_ul56', 0: 'count'})

    #print(to_keep_counts)
    to_keep_counts.to_csv('./'+'to_keep_counts56.csv')

if __name__ == '__main__':
    main()

