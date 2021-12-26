from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('FATAL')

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv",
                                     "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv",
                                    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')


def my_input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3
)

classifier.train(
    input_fn=lambda: my_input_fn(train, train_y),
    steps=5000
)

eval_result = classifier.evaluate(input_fn=lambda: my_input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
print(eval_result)


def input_fn_pred(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


my_features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict_dict = {}

print('enter numbers as prompted')
for feature in my_features:
    valid = False
    val = "woops"
    while not valid:
        val = input(feature + ": ")
        if val.isdigit():
            valid = True
    predict_dict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn_pred(predict_dict))
for pred in predictions:
    class_id = pred['class_ids'][0]
    probability = pred['probabilities'][class_id]

    print('\nPrediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))
