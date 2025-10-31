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
import subprocess
import tensorflow as tf

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

def main():

    global X_train, y_train, X_test, y_test
    
    num_epochs = 50
    num_batch = 64
    
    '''obtain proper form of train test data'''
    #load json file which are all game examples with ground truth
    
    #create data on the fly
    
    games = SetCoverDataGenerator()
    #convert to dictionary
    games_dic = json.loads(games)
    games_df = dict2df(games_dic)
    
    #games_df = pd.read_csv('./'+'fixdata.csv',index_col=0) #used for one-time generated fixed data
    #games_df = pd.read_csv('./'+'vardata56.csv',index_col=0) #used for one-time generated various types of data and staack different data together
    #games_df.to_csv('./'+'games_df.csv')
    df = pd.get_dummies(games_df) #onehot encoding just in case any categorical data show up
    #seperate attributes and the label, columns with optimal key words is label.
    #df.to_csv('./' + 'original_game.csv')
    
    description = df.describe()

    print(description)

    X = df.drop(columns=[col for col in df.columns if 'optimal' in col]) #feature data
    X.to_csv('./' + 'X.csv')
    y = df[[col for col in df.columns if 'optimal' in col]] #labels 
    #print entire dataset
    y.to_csv('./' + 'y.csv')
    #print('\n\n original dataset:\n',df)

    print("data size:", len(df))
    
    #split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    #further split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=21)


    best_model=keras_tune(X_train, X_val, y_train, y_val)
    best_model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch, validation_data = (X_val, y_val),verbose=0)
    best_model.save("submodel")#save model


    loaded_model = tf.keras.models.load_model("submodel")
    '''
    #print load model info
    loaded_model.summary()

    #get the model's weights
    model_weights = loaded_model.get_weights()
    
    #print or inspect the weights
    print('layer weights')
    for layer_weights in model_weights:
        print(layer_weights)
    '''
    y_pred = loaded_model.predict(X_test)
    test_loss = loaded_model.evaluate(X_test, y_test)
    print('MSE', test_loss)

    #print('predicted: %s' % y_pred[0])
    #convert matrix to panda data frame with column names and match indices
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)
    #print ('\n\n','X_test:\n', X_test,'\n\n','y_test regression:\n', y_test,'\n\n','y_pred regression:\n', y_pred_df,'\n\n')

    #post processing to convert the regression values back to 0 and 1
    df_predwx = pd.concat([X_test, y_pred_df], axis=1)#concact X_test and y_pred_df
    df_testwx = pd.concat([X_test, y_test], axis=1)#concact X_test and y_test
    df_testwx = find_sum_r(df_testwx)#find and create a column with sum_R using the original test labels with x
    #df_testwx.to_csv('./'+'correct_answer.csv')
    
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


