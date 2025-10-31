'''keepvswelfare.py
This is used to test the numbers to keep.
see what is the optimal number for number to keep vs welfare
this code specifically controlled 
'''

'''import json and pand'''
import json #data structure as key value pairs and array
import pandas as pd #import python data structure 

'''import user defined modules'''
from datagen_var import SetCoverDataGenerator #dataset generator from https://github.com/joshuahseaton/SubmodOptDatasetGen.git
from dict2df import dict2df #convert data generated from datagen.py

from keras_tune import keras_tune #import keras tuner, return the best tuned model
from rad_seq import rad_seq #import random sequential processing
from seq_greedy import seq_greedy #import sequential greedy
from blind_greedy import blind_greedy #import blind greedy
from blind_postps import blind_postps #import blind post processing, convert the regression matrix to 0 and 1 by picking the max value between same agents and calculate wellfare
from blind_rad import blind_rad
from seq_postps import seq_postps
from find_sum_r import find_sum_r #find sum of picked R from optimal choices
from welfare import welfare #plot welfare

#mlp for multi-output regression
from sklearn.model_selection import train_test_split #import scikit-learn to split train and test sets
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

def main():

    # Load the array from the file
    to_keep_list = np.load('to_keep_list.npy')

    print(to_keep_list)

    
    #array to save all average welfare values for plotting purposes
    welfare_values_rad_seq = []
    welfare_values_seq_greedy = []
    welfare_values_blind_greedy = []
    welfare_values_blind_nn = []
    welfare_values_seq_nn = []
    welfare_values_blind_rad = []

    for i in range(0, 72): 
        print(i)
        #load X_test DataFrame
        X_test = pd.read_csv(f'individual_X_test_{i}.csv')
        #load y_test DataFrame
        y_test = pd.read_csv(f'individual_y_test_{i}.csv')
        loaded_model = tf.keras.models.load_model("submodel")

        y_pred = loaded_model.predict(X_test)
        #print('predicted: %s' % y_pred[0])
        #convert matrix to panda data frame with column names and match indices
        y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

        #print ('\n\n','X_test:\n', X_test,'\n\n','y_test regression:\n', y_test,'\n\n','y_pred regression:\n', y_pred_df,'\n\n')

        #post processing to convert the regression values back to 0 and 1
        df_predwx = pd.concat([X_test, y_pred_df], axis=1)#concact X_test and y_pred_df
        df_testwx = pd.concat([X_test, y_test], axis=1)#concact X_test and y_test

        df_testwx = find_sum_r(df_testwx)#find and create a column with sum_R using the original test labels with x
        
        #call random sequential processing
        X_radseq = rad_seq(X_test)
        welfare_values_rad_seq.append(welfare(X_radseq, df_testwx))

        #call sequential greedy
        X_sqgreedy = seq_greedy(X_test)
        welfare_values_seq_greedy.append(welfare(X_sqgreedy, df_testwx))

        #call blind rad
        X_bldrad= blind_rad(X_test)
        welfare_values_blind_rad.append(welfare(X_bldrad, df_testwx))

        #call blind greedy
        X_bldgreedy = blind_greedy(X_test)
        welfare_values_blind_greedy.append(welfare(X_bldgreedy, df_testwx))

        #call blind nn processing
        df_blindnn = blind_postps(df_predwx)
        df_blindnn = find_sum_r(df_blindnn)
        welfare_values_blind_nn.append(welfare(df_blindnn, df_testwx))

        #call sequential nn processing
        df_seqnn = seq_postps(df_predwx)
        df_seqnn = find_sum_r(df_seqnn)
        welfare_values_seq_nn.append(welfare(df_seqnn, df_testwx))


    #plot all welfare values together
    plt.plot(to_keep_list, welfare_values_rad_seq, label='random sequential')
    plt.plot(to_keep_list, welfare_values_seq_greedy, label='sequential greedy')
    plt.plot(to_keep_list, welfare_values_blind_greedy, label='blind greedy')
    plt.plot(to_keep_list, welfare_values_blind_nn, label='blind NN')
    plt.plot(to_keep_list, welfare_values_blind_rad, label='blind rad')
    plt.plot(to_keep_list,welfare_values_seq_nn, label='sequential NN')
    plt.xlabel('to_keep')
    plt.ylabel('ave_welfare')
    plt.legend()

    plt.savefig('slicewelfare.png')
    #plt.show()
    plt.clf()

if __name__ == '__main__':
    main()


