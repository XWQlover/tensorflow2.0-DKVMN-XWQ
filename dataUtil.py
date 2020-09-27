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

        self.train = self.seq.sample(frac=0.8)
        self.test = self.seq[~self.seq.index.isin(self.train.index)]


    def datasetReturn(self, data, shuffle=None, batch_size=32, val_data=None):
        dataset = tf.data.Dataset.from_generator(lambda: data, output_types=(tf.int32, tf.int32, tf.int32))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)

        MASK_VALUE = -1
        dataset = dataset.padded_batch(
            batch_size=50,
            padding_values=(MASK_VALUE, MASK_VALUE, MASK_VALUE),
            padded_shapes=([None], [None], [None]),
            drop_remainder=True
        )

        return dataset