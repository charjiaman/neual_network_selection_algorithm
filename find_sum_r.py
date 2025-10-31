'''
function name: find_sum_r.py 

find_sum_r.py iterate through optimal.Px.Rx, store the Rx where optimal.Px.Rx value is 1 in the series, find the Rx value in the original Rx column, sum them to sum_r column. each R value can only be added once to the final sum_r.

optimal.P0.R0	optimal.P0.R1	optimal.P0.R2  optimal.P1.R0	optimal.P1.R1	optimal.P1.R2
       1	           0	           0              1	               0	           0
  
find R0 from optimal.P0.R0, find R0 from optimal.P1.R0, go back to original R0 column, add R0 value to sum_r, in this case R0 is added only once to sum_r though both P0 and P1 picked R0.

note in this find_sum_r, code is not checking if action.Px.Rx is !0. infeasible actions should be taken care in the post processing before find_sum_r. blind_postps_v1 will check action.Px.Rx is !0. See ref blind_postps_v1.
'''
import copy
def find_sum_r(df_wx):
    df_wxcopy = copy.deepcopy(df_wx)
    for index, row in df_wxcopy.iterrows():
        unique_r_values = set() #create a new, unordered collection of unique elements. 
        for col in row.index:
            if col.startswith('optimal.') and row[col] == 1: 
            #specified column (col) of the current row (row)
                r_value = col.split('.')[2] #access the third element ["optimal", "P0", "R2"]
                if r_value.startswith('R'):
                    unique_r_values.add(r_value) 
        #access R columns and calculate sum
        #sum_r = row[unique_r_values].sum()
        sum_r = row[list(unique_r_values)].sum()
        #create a new column with the sum of R values for the current row
        df_wxcopy.at[index, 'sum_r'] = sum_r

    return df_wxcopy #return a table with previoues feature X and the sum of unique Rs.