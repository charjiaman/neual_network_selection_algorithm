'''
function name: main.py

main.py file is to generate data, clean data and call mutiple classifiers to evaluate data
this main contains simple regression and commented decision tree sample code
'''

'''import json and pand'''
import json #data structure as key value pairs and array
import pandas as pd #import python data structure 

'''import user defined modules'''
from datagen import SetCoverDataGenerator #dataset generator from https://github.com/joshuahseaton/SubmodOptDatasetGen.git
from dict2df import dict2df #convert data generated from datagen.py

from keras_tune import keras_tune #import keras tuner, return the best tuned model
from rad_seq import rad_seq #import random sequential processing
from seq_greedy import seq_greedy #import sequential greedy
from blind_greedy import blind_greedy #import blind greedy
from blind_postps import blind_postps #import blind post processing, convert the regression matrix to 0 and 1 by picking the max value between same agents and calculate wellfare
from seq_postps import seq_postps
from blind_rad import blind_rad
from find_sum_r import find_sum_r #find sum of picked R from optimal choices
from welfare_plt import welfare_plt #plot welfare

#mlp for multi-output regression
from sklearn.model_selection import train_test_split #import scikit-learn to split train and test sets

import tensorflow as tf
import subprocess

'''
#for windows only

#embed #1. rmdir /s /q mlsubtuner in python code to clean kera tuner cache
#directory path to remove (replace with your desired path)
directory_to_remove = 'submodel'
try:
    #run the rmdir command to remove the directory
    subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', directory_to_remove], check=True)
    print(f"Directory '{directory_to_remove}' removed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
'''
def main():

    global X_train, y_train, X_test, y_test
    
    num_epochs = 50
    num_batch = 64

    X_train = pd.read_csv('./'+'stacked_X_train.csv')
    y_train = pd.read_csv('./'+'stacked_y_train.csv')
    X_val = pd.read_csv('./'+'stacked_X_val.csv')
    y_val = pd.read_csv('./'+'stacked_y_val.csv')
    X_test = pd.read_csv('./'+'stacked_X_test.csv')
    y_test = pd.read_csv('./'+'stacked_y_test.csv')
        

    best_model=keras_tune(X_train, X_val, y_train, y_val)
    best_model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch, validation_data = (X_val, y_val),verbose=0)
    best_model.save("submodel")#save model

    loaded_model = tf.keras.models.load_model("submodel")

    y_pred = loaded_model.predict(X_test)
    test_loss = loaded_model.evaluate(X_test, y_test)
    print('MSE', test_loss)

    #print('predicted: %s' % y_pred[0])
    #convert matrix to panda data frame with column names and match indices
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)
    #print ('\n\n','X_test:\n', X_test,'\n\n','y_test regression:\n', y_test,'\n\n','y_pred regression:\n', y_pred_df,'\n\n')
    y_pred_df.to_csv('./preddf.csv')
    #post processing to convert the regression values back to 0 and 1
    df_predwx = pd.concat([X_test, y_pred_df], axis=1)#concact X_test and y_pred_df
    df_testwx = pd.concat([X_test, y_test], axis=1)#concact X_test and y_test
    df_testwx = find_sum_r(df_testwx)#find and create a column with sum_R using the original test labels with x
    df_testwx.to_csv('./'+'correct_answer.csv')
    
    #call blind random processing
    X_blindrad = blind_rad(X_test)
    welfare_plt(X_blindrad, df_testwx , 'pink', 'welfare-blind-random')
    #X_blindrad.to_csv('./'+'blindrad_sum.csv')    
    
    #call random sequential processing
    X_radseq = rad_seq(X_test)
    welfare_plt(X_radseq, df_testwx , 'purple', 'welfare-random-sequential')
    #X_radseq.to_csv('./'+'radseq_sum.csv')    
    
    #call sequential greedy
    X_seqgreedy = seq_greedy(X_test)
    welfare_plt(X_seqgreedy, df_testwx , 'brown', 'welfare-sequential-greedy')
    #X_seqgreedy.to_csv('./'+'seqgd_sum.csv')
 
    #call blind greedy
    X_blindgreedy = blind_greedy(X_test)
    welfare_plt(X_blindgreedy, df_testwx , 'yellow', 'welfare-blind-greedy')
    #X_blindgreedy.to_csv('./'+'blindgd_sum.csv')
    
    #call sequential nn processing
    df_seqnn = seq_postps(df_predwx)
    df_seqnn = find_sum_r(df_seqnn)
    welfare_plt(df_seqnn, df_testwx , 'green', 'welfare-sequential-nn')
    #df_seqnn.to_csv('./'+'seqnn_sum.csv')

    #call blind nn processing
    df_blindnn = blind_postps(df_predwx)
    df_blindnn = find_sum_r(df_blindnn)
    welfare_plt(df_blindnn, df_testwx , 'blue', 'welfare-blind-nn')
    #df_blindnn.to_csv('./'+'blindnn_sum.csv')
    
if __name__ == '__main__':
    main()


