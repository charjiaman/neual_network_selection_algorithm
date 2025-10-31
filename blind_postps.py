'''
function name: blind_postps.py - complete

blind_postps.py performs post processings on the predicted regression results and convert to 0 and 1 labels:

blind_postps.py is considered as blind where the post processing agent can pick the resources that previous agents have already picked.

if action.Px.Rx == 0 then the optimal.Px.Rx will be set to -math.inf.
in this case, max (optimal.Px.Rx) will not consider the -math.inf

this code will limit the infeasible before apply max to the regression value.
see seq_postps_v1.py for reference where the infeasible optimal can be chosen but will not be considered in the sum_r

blind_postps.py performs post processings on the predicted regression results and convert to 0 and 1 labels:

for each groups of optimal columns with same Px, e.g. optimal.P1.R1, optimal.P1.R2... find the column with max value, set the largest column to 1 and all the others to 0.

since this is blind, both P1 and P0 can choose R0 in the following case. But R0 will only be added once in the find_sum_r.py

    optimal.P0.R0	optimal.P0.R1	optimal.P0.R2   optimal.P1.R0	optimal.P1.R1	optimal.P1.R2
          0.8	          0.1	          0.3             0.5	          0.2	          0.2

    convert to: 

    optimal.P0.R0	optimal.P0.R1	optimal.P0.R2   optimal.P1.R0	optimal.P1.R1	optimal.P1.R2
          1	               0	           0               1	           0	           0

'''
import math
import copy
import pandas as pd

#use P0 as example
def blind_postps(df_predwx):
    df_predwxcopy = copy.deepcopy(df_predwx)
    #df_predwxcopy.to_csv('./'+'blindnn_init.csv')


    #combined_data = pd.DataFrame()
    df_predwxcopy = copy.deepcopy(df_predwx)
    #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)

    #find the action.Px.Rx = 0 and mark optimal.Px.Rx to -inf indicating infeasible
    for index, row in df_predwxcopy.iterrows():
        for col in row.index:
            if col.startswith('action.') and row[col] == 0: 
            #specified column (col) of the current row (row)
                px_value = col.split('.')[1]  #extract Px
                r_value = col.split('.')[2]  #extract Rx
            #check if the corresponding 'action.Px.Rx' column is not 0
            #if the action.Px.Rx is 0 but the optimal.Px.Rx is 1 then R is not feasible for choice 
                action_col = f'action.{px_value}.{r_value}'
                if action_col in row.index and row[action_col] == 0:
                    opti_col = f'optimal.{px_value}.{r_value}'
                    df_predwxcopy.at[index, opti_col] = -math.inf #marked infeasible success
    #df_predwxcopy.to_csv('blindnn_mid.csv')  
    #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)
         
    #iterate through unique P values, set 1 to the max optimal.P0.Rx
    unique_p_values = df_predwxcopy.columns.to_series().str.extract(r'optimal\.(\w+)\.')[0].dropna().unique()
    for index, row in df_predwxcopy.iterrows(): 
        for p_value in unique_p_values:
            px_columns = df_predwxcopy.columns[df_predwxcopy.columns.str.startswith(f'optimal.{p_value}.')]
            max_col = df_predwxcopy[px_columns].idxmax(axis=1)
            #print('agent', p_value,'\n', 'max col','\n',max_col)
            # Check if all values in px_columns are -inf
            if (df_predwxcopy.loc[index, px_columns] == -math.inf).all():
                # If all values are -inf, set everything to 0
                df_predwxcopy.loc[index, px_columns] = 0
            else:
                max_col = df_predwxcopy.loc[index, px_columns].idxmax()
                df_predwxcopy.loc[index, px_columns] = 0
                df_predwxcopy.at[index, max_col] = 1
    #df_predwxcopy.to_csv('blindnn_final.csv')
    #combined_data = pd.concat([combined_data, df_predwxcopy], ignore_index=False)
    #combined_data.to_csv('./'+ 'blindnn_combinedall.csv')
    return df_predwxcopy


