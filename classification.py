import tensorflow as tf
import pandas as pd

BATCH_SIZE = 256

# this example uses the iris flowers dataset
CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]

TRAIN_PATH = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)

TEST_PATH = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)



def input_fn(features, labels, training = True, batch_size = BATCH_SIZE):
    # convert inputs into tf.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # shuffle if in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def main():
    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0)

    print(train.head())

    y_train  = train.pop("Species")
    y_test   = test.pop("Species")

    # make feature columns
    feature_columns = []
    for key in train.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # create model

    # the main PREFAB (if you like) classifiers in TF are
    #   DNNClassifier (deep neural network)
    #   LinearClassifier (linear regression)
    model = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # two hidden layers of 30 and 10 nodes 
        hidden_units = [30, 10],
        # model must choose between 3 classes
        n_classes = 3
    )

    model.train(
        input_fn=lambda: input_fn(train, y_train, training=True),
        steps = 5000
    )

    result = model.evaluate(
        input_fn=lambda: input_fn(test, y_test, training=False)
    )

    print(f"\nTest set accuracy: {result.accuracy}")

if __name__ == "__main__":
    main()