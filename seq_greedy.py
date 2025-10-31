'''
function name: seq_greedy.py - complete

seq_greedy.py is considered as sequential where the post processing agent cannot pick the resources that has been chosen by previous agents.

seq_greedy start from each agent, pick the R with largest value that is avaialble in action.Px.Rx, add it to the sum_r. After R being chosen, R is no longer available, then go to the next agent, repeat.

	 P0	           P1	      R0	            R1	         R2	
     1             1	     0.8   	           0.9           0.6


action.P0.R0	action.P0.R1	action.P0.R2	action.P1.R0	action.P1.R1	action.P1.R2
	1	              1	               1	         0	             1	              1

P0 can choose R0, R1 and R2, P0 pick R1. resources left {R0, R2}
P1 can choose R1 and R2, since R1 has been picked by P0,  P1 pick R2.

'''
import copy
import pandas as pd

def seq_greedy(X_test):
    #combined_data = pd.DataFrame()
    X_testcopy = copy.deepcopy(X_test)
    #combined_data = pd.concat([combined_data, X_testcopy], ignore_index=False)

    #X_testcopy.to_csv('./'+'seqgd_innit.csv')
    #iterate through unique P values in action.Px.Rx
    unique_p_values = X_testcopy.columns.to_series().str.extract(r'action\.(\w+)\.')[0].dropna().unique()
    X_testcopy.loc[:, 'sum_r'] = 0.0
    #iterate each row
    for index, row in X_testcopy.iterrows():#iter each row, 'row' contains all data for the current row
        #print('index',index)
        picked_r = set() #sets of unique elements.
        for p_value in unique_p_values: #under row, for each agent P
            px_columns = X_testcopy.columns[X_testcopy.columns.str.startswith(f'action.{p_value}.')]#get action columns for each P: e.g. P0 - action.P0.R0	action.P0.R1	action.P0.R2
            agent_r = set()
            #iterate through action.P0.Rxs
            for col in px_columns:
                if row[col] == 1: #if the agent has resource that it can pick
                    R_col = col.split('.')[2] #get Rx names that each agent can choose
                    if R_col.startswith('R') and R_col not in picked_r:
                        agent_r.add(R_col) #unique R under each agent action 
            #print(agent_r,'R available for', p_value)
            #if there are unique R values
            if agent_r is not set():
                max_r_value = 0.0 #initialize max_r_value to 0.0              
                max_r_col = None #initialize max_r_col name to None
                #iterate through unique R values and find the largest value
                for R_col in agent_r:
                    column_name = f'{R_col}'
                    if column_name in row.index:
                        value = row[column_name]
                        if value > max_r_value:
                            max_r_value = value
                            max_r_col = R_col
            #print('max_R',max_r_value,'for agent',p_value)
            if max_r_value != 0.0:   
                #add the largest 'R' value to 'sum_r'
                X_testcopy.at[index, 'sum_r'] += max_r_value 
                #set the picked_r value in the original Rx to 0
                X_testcopy.at[index, f'{max_r_col}'] = 0.0
                #add picked R to picked_r set
                picked_r.add(max_r_col)       
    #X_testcopy.to_csv('./'+'seqgd_final.csv')
    #combined_data = pd.concat([combined_data, X_testcopy], ignore_index=False)
    #combined_data.to_csv('./'+'seqgd_combinedall.csv')
    return X_testcopy



