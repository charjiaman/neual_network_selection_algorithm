'''
function name: seq_postps.py

se1_postps.py performs post processings on the predicted regression results and convert to 0 and 1 labels:

se1_postps.py is considered as sequential where the post processing agent cannot pick the resources that has been chosen by previous agents.

if action.Px.Rx == 0 then the optimal.Px.Rx will be set to -math.inf.
in this case, max (optimal.Px.Rx) will not consider the -math.inf

this code will limit the infeasible before apply max to the regression value.
see seq_postps_v1.py for reference where the infeasible optimal can be chosen but will not be considered in the sum_r

se1_postps.py performs post processings on the predicted regression results and convert to 0 and 1 labels:

for each groups of optimal columns with same Px, e.g. optimal.P1.R1, optimal.P1.R2... find the column with max value, set the largest column to 1 and all the others to 0.


    optimal.P0.R0	optimal.P0.R1	optimal.P0.R2   optimal.P1.R0	optimal.P1.R1	optimal.P1.R2
          0.8	          0.1	          0.3             0.5	          0.2	          0.2

    P0 picked R0, mark optimal.P0.R0 to 1, mark optimal.P1.R0 to -math.inf 

    optimal.P0.R0	optimal.P0.R1	optimal.P0.R2   optimal.P1.R0	optimal.P1.R1	optimal.P1.R2
          1	             0.1	          0.3           -math.inf	       0.2	          0.2

    P1 picked R1

    optimal.P0.R0	optimal.P0.R1	optimal.P0.R2   optimal.P1.R0	optimal.P1.R1	optimal.P1.R2
          1	               0	           0               0	           1	           0

'''

import math
import copy
import csv
import pandas as pd

#use P0 as an example
def seq_postps(df_predwx):
    
    #df_predwx.to_csv('./' + 'ini.csv')
    #print(df_predwx)
    #find the action.Px.Rx = 0 and mark optimal.Px.Rx to -inf indicating infeasible
    #combined_data = pd.DataFrame()
    df_predwxcopy = copy.deepcopy(df_predwx)
    #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)

    #df_predwxcopy.to_csv('./'+'seqnn_innit.csv')

    for index, row in df_predwxcopy.iterrows():
        for col in row.index:
            if col.startswith('action.') and row[col] == 0:
                #specified column (col) of the current row (row)
                px_value = col.split('.')[1]  #extract Px
                r_value = col.split('.')[2]  #extract Rx
                #check if the corresponding 'action.Px.Rx' column is not 0
                #if the action.Px.Rx is 0 but the optimal.Px.Rx is 1 then R is not feasible for choice
                opti_col = f'optimal.{px_value}.{r_value}'
                df_predwxcopy.at[index, opti_col] = -math.inf  #marked infeasible success

    #df_predwxcopy.to_csv('./'+'seqnn_mid.csv')
    #print(df_predwxcopy)
    #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)
    #iterate through unique P values, set 1 to the max optimal.P0.Rx
    unique_p_values = df_predwxcopy.columns.to_series().str.extract(r'optimal\.(\w+)\.')[0].dropna().unique()

    for index, row in df_predwxcopy.iterrows():  #iterate each row, 'row' contains all data for the current row

        picked_r = set()
        selected_p_values = []
        opti_col = set()

        for p_value in unique_p_values:  # For each unique p value
            selected_p_values.append(p_value)
            #other p value
            other_p_values = [p for p in unique_p_values if p not in selected_p_values]
            #print('other p', other_p_values)
            px_columns = df_predwxcopy.columns[df_predwxcopy.columns.str.startswith(f'optimal.{p_value}.')] #find all optimal.P0.Rx

            #check if all values in px_columns are -inf
            if (df_predwxcopy.loc[index, px_columns] == -math.inf).all():
                #if all values are -inf, set everything to 0
                df_predwxcopy.loc[index, px_columns] = 0
            else:
                max_col = df_predwxcopy.loc[index, px_columns].idxmax()
                df_predwxcopy.loc[index, px_columns] = 0
                df_predwxcopy.at[index, max_col] = 1

                #('index', index,'agent', p_value, 'current max', max_col)
                picked_r.add(max_col.split('.')[2])
                #print('picked r', picked_r)
                #print('px_xolumna', px_columns)
                #print('max_col', max_col)
            
            if picked_r:
                for r in picked_r:
                    for px_value in other_p_values:
                        opti_col = f'optimal.{px_value}.{r}'
                        #print('index', index, opti_col,  'should be inked with -inf')
                        df_predwxcopy.at[index, opti_col] = -math.inf
                #print(df_predwxcopy)
                        #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)

    #print(df_predwxcopy)
    #df_predwxcopy.to_csv('./'+'seqnn_final.csv')
    #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)
    #combined_data.to_csv('./'+'seqnn_process.csv')   
    return df_predwxcopy


