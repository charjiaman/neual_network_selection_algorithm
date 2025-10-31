'''
function name: genfixdata.py

genfixdata.py generate data and export to csv so the training can use a fixed set of data for innitial stage of training.
'''

import json #data structure as key value pairs and array
import pandas as pd #import python data structure 

'''import user defined modules'''
from datagen import SetCoverDataGenerator #dataset generator from https://github.com/joshuahseaton/SubmodOptDatasetGen.git
from dict2df import dict2df #convert data generated from datagen.py

def main():

    '''obtain proper form of train test data'''
    #load json file which are all game examples with ground truth
    games = SetCoverDataGenerator()
    #convert to dictionary
    games_dic = json.loads(games)
    games_df = dict2df(games_dic)

    #print('\n\n original game_df:\n',games_df)
    games_df.to_csv('./'+'fixdata.csv')#output games_df to csv files

if __name__ == '__main__':
    main()


