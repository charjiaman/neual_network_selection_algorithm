'''
function: welfare.py

create a seperate dataframe df_well to store the welfare where welfare = sum of R selected by prediction/sum of R from the original test labels.
return average welfare
'''

import pandas as pd #import python data structure   

def welfare(df_predwx, df_testwx):

    df_well = pd.DataFrame()
    df_well['welfare'] = df_predwx['sum_r']/df_testwx['sum_r']#welfare is equal to prediction value/original ground truth
    #calculate mean
    average_value = df_well['welfare'].mean()

    return average_value