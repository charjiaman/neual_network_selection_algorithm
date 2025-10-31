'''
function name: rad_seq.py - complete

rad_seq.py is considered as sequential where the post processing agent cannot pick the resources that has been chosen by previous agents.

rad_seq start from each agent, pick any R value that is avaialble in action.Px.Rx, add it to the sum_r. After R being chosen, R is no longer available, then go to the next agent, repeat.

	 P0	           P1	       R0	         R1	         R2	
     1             1	       0.8   	     0.7           0.6


action.P0.R0	action.P0.R1	action.P0.R2	action.P1.R0	action.P1.R1	action.P1.R2
	1	              1	               1	         0	             1	              1

P0 can choose R0, R1 and R2, P0 pick R1. resources left {R0, R2}
P1 can choose R1 and R2, since R1 has been picked by P0,  P1 pick R2.

'''
import random
import copy
import pandas as pd

def rad_seq(X_test):
    X_testcopy = copy.deepcopy(X_test)
    #combined_data = pd.DataFrame()
    #X_testcopy.to_csv('./'+'radseq_init.csv')
    #combined_data = pd.concat([combined_data, X_testcopy], ignore_index=False)
    #iterate through unique P values in action.Px.Rx
    unique_p_values = X_testcopy.columns.to_series().str.extract(r'action\.(\w+)\.')[0].dropna().unique()
    X_testcopy.loc[:, 'sum_r'] = 0.0
    #iterate each row
    random.shuffle(unique_p_values)
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
            if not agent_r:
                pass
            else:
                random_resource = random.choice(list(agent_r))
                column_name = f'{random_resource}'
                #get the corresponding r_value from the selected column
                r_value = row.get(column_name, 0.0) #If column_name does not exist in the row's index, it will return the default value 0.0, and that value will be assigned to r_value.
                if r_value != 0.0:
                    X_testcopy.at[index, 'sum_r'] += r_value
                    X_testcopy.at[index, column_name] = 0.0
                    picked_r.add(random_resource)
                    #print (p_value, 'picked', random_resource)
    #X_testcopy.to_csv('./'+'radseq_final.csv')
    #combined_data = pd.concat([combined_data, X_testcopy], ignore_index=False)
    #combined_data.to_csv('./'+'radseq_combinedall.csv')
    return X_testcopy



