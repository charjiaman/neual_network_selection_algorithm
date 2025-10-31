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
from seq_postps import seq_postps
from find_sum_r import find_sum_r #find sum of picked R from optimal choices
from welfare import welfare #plot welfare

#mlp for multi-output regression
from sklearn.model_selection import train_test_split #import scikit-learn to split train and test sets

import matplotlib.pyplot as plt

def main():

    #array to save all average welfare values for plotting purposes
    welfare_values_rad_seq = []
    welfare_values_seq_greedy = []
    welfare_values_blind_greedy = []
    welfare_values_blind_nn = []
    welfare_values_seq_nn = []

    ordata_sizes = []
    rmdata_sizes = []
    spdadta_sizes = []
    train_sizes = []  # List to store training set sizes
    val_sizes = []    # List to store validation set sizes
    test_sizes = []   # List to store test set sizes
    


    global X_train, y_train, X_test, y_test
    num_epochs = 50
    num_batch = 64

    NUM_AGENTS = 8
    NUM_RESOURCES = 2 * NUM_AGENTS
    NUM_GAMES = 1000
    keep_ul = 49
    #actionsto_remove = NUM_AGENTS * NUM_RESOURCES - to_keep
    #range does not include the right )
    for to_keep in range(2, keep_ul): 
        print('loop:', to_keep)
        '''obtain proper form of train test data'''
        #load json file which are all game examples with ground truth
        games = SetCoverDataGenerator(NUM_RESOURCES, NUM_AGENTS, to_keep, NUM_GAMES)

        '''obtain proper form of train test data'''
        #load json file which are all game examples with ground truth
        #convert to dictionary
        games_dic = json.loads(games)
        games_df = dict2df(games_dic)
        '''create data on the fly'''
        df = pd.get_dummies(games_df) #onehot encoding just in case any categorical data show up
        #seperate attributes and the label, columns with optimal key words is label.
        print("original data size:", len(df))  

        target_training_size = 1000

        # Calculate the number of samples to remove
        samples_to_remove = len(df) - target_training_size
        ordata_sizes.append(len(df))
        # Randomly remove samples without changing the training size
        if samples_to_remove > 0:
            df = df.sample(frac=1, random_state=42).iloc[samples_to_remove:]
        spdadta_sizes.append(len(df))
        print("sample data size:", len(df))  

        X = df.drop(columns=[col for col in df.columns if 'optimal' in col]) #feature data
        y = df[[col for col in df.columns if 'optimal' in col]] #labels 
        #print entire dataset
        #print('\n\n original dataset:\n',df)
        
        #split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #further split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
        print("training set size:", len(X_train))
        print("validation set size:", len(X_val))
        print("test set size:", len(X_test))
        best_model = keras_tune(X_train, X_val, y_train, y_val)
        best_model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch, validation_data = (X_val, y_val),verbose=0)
        test_loss = best_model.evaluate(X_test, y_test)
        print('MSE', test_loss)
        y_pred = best_model.predict(X_test)

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

       
        rmdata_sizes.append(samples_to_remove)
        train_sizes.append(len(X_train))
        val_sizes.append(len(X_val))
        test_sizes.append(len(X_test))

    # Plot the training, validation, and test set sizes
    plt.figure(1)
    plt.plot(train_sizes, label='training set size', linestyle='--')
    plt.plot(val_sizes, label='validation set size', linestyle='--')
    plt.plot(test_sizes, label='test set size', linestyle='--')
    plt.plot(rmdata_sizes, label='removed data size', linestyle='--')
    plt.plot(ordata_sizes, label='original data size', linestyle='--')
    plt.plot(spdadta_sizes, label='sample data size', linestyle='--')  
    plt.xlabel('to_keep')
    plt.ylabel('size')
    plt.legend()
    plt.savefig('keepvssize.png')
    plt.figure(2)
    #plot all welfare values together
    plt.plot(welfare_values_rad_seq, label='random sequential')
    plt.plot(welfare_values_seq_greedy, label='sequential greedy')
    plt.plot(welfare_values_blind_greedy, label='blind greedy')
    plt.plot(welfare_values_blind_nn, label='blind NN')
    plt.plot(welfare_values_seq_nn, label='sequential NN')
    plt.xlabel('to_keep')
    plt.ylabel('welfare')
    plt.legend()

    plt.savefig('keepvswelfare.png')
    plt.show()
if __name__ == '__main__':
    main()


