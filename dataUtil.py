import pandas as pd
import numpy as np
import tensorflow as tf
class AssismentData():
    def __init__(self):
        self.data = pd.read_csv("C://Users//Administrator//Desktop//data//2015_100_skill_builders_main_problems.csv")

        self.data.dropna()

        self.data["user_id"], _ = pd.factorize(self.data["user_id"])
        self.data["sequence_id"], _ = pd.factorize(self.data["sequence_id"])
        self.data["skills"] = self.data.apply(lambda x: x.sequence_id * 2 if x.correct == 0.0 else x.sequence_id * 2 + 1, axis=1)

        self.data = self.data.drop(columns="log_id", axis=1)

        self.data = self.data.groupby("user_id").filter(lambda q: len(q) > 1).copy()

        self.seq = self.data.groupby('user_id').apply(
            lambda r: (
                r["sequence_id"].values,
                r['skills'].values,
                r['correct'].values
            )
        )

    def datasetReturn(self, shuffle=None, batch_size=32, val_data=None):

        dataset = tf.data.Dataset.from_generator(lambda: self.seq, output_types=(tf.int32, tf.int32, tf.int32))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)
        # mask
        MASK_VALUE = -1
        # padd
        dataset = dataset.padded_batch(
            batch_size=3,
            padding_values=( MASK_VALUE,MASK_VALUE, MASK_VALUE),
            padded_shapes=( [None], [None], [None]),
            drop_remainder=True
        )
        i = 0
        for l in dataset.as_numpy_iterator():
            i += 1
        # split
        test_size = int(np.ceil(i * 0.2))
        train_size = i - test_size
        val_size = int(np.ceil(i * 0.2))
        train_size = train_size - val_size

        test_data = dataset.take(test_size)
        dataset = dataset.skip(test_size)

        val_data = dataset.take(val_size)
        dataset = dataset.skip(val_size)

        return dataset, test_data, val_data