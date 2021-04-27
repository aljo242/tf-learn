import tensorflow as tf
import matplotlib.pyplot as plt     # best visualization lib in the game
import numpy as np                  # optimized array computation
import pandas as pd                 # dataframe representation
from six.moves import urllib        # 

BATCH_SIZE = 32
NUM_EPOCHS = 10

plotX = [1, 2, 2.5, 3, 4]
plotY = [1, 4, 7, 9, 15]

def makeInputFN(data_df, label_df, num_epochs = NUM_EPOCHS, shuffle = True, batch_size = BATCH_SIZE):
    def input_function(): # Inner function to be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000) # randomize the order of data
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function # return function obj



def plotData(x, y):
    plt.plot(x, y, 'ro')
    plt.axis([0, 6, 0, 20])

def npFit():
    print("\nusing numpy to find line of best fit...")
    plotData(plotX, plotY)
    plt.plot(np.unique(plotX), np.poly1d(np.polyfit(plotX, plotY, 1))(np.unique(plotX)))
    plt.show()

def tfFit():
        print("\nusing sklearn to find line of best fit...")

        # get sets
        dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
        dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
        y_train = dftrain.pop('survived')
        y_eval = dfeval.pop('survived')

        print(dftrain.describe())

        CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
        NUMERIC_COLUMNS = ['age', 'fare']

        feature_columns = []
        for feature_name in CATEGORICAL_COLUMNS:
            vocab = dftrain[feature_name].unique() # get a list of all unique values for the given column
            feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

        for feature_name in NUMERIC_COLUMNS:
            feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

        train_input_fn = makeInputFN(dftrain, y_train)
        eval_input_fn = makeInputFN(dfeval, y_eval, num_epochs = 1, shuffle = False)

        linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)

        linear_est.train(train_input_fn)                # train 
        result = linear_est.evaluate(eval_input_fn)     # test
        print("\n\n")
        print(result)



def main():
    npFit()

    tfFit()

if __name__ == "__main__":
    main()