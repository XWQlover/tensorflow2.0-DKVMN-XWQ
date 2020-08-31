from dataUtil import  AssismentData
import tensorflow as tf
from DKVMN import DKVMN

ass = AssismentData()
train_data,test_data,val_data = ass.datasetReturn()
val_log = 'log/val'
train_loss_log = 'log/train'
summary_writer = tf.summary.create_file_writer(val_log)

dkvmn = DKVMN()
bc = tf.metrics.BinaryCrossentropy()
auc = tf.metrics.AUC()
vauc = tf.metrics.AUC()
optimizer = tf.optimizers.Adam()

def test_one_step(skillid, skill_correctness, correctness):
    probility = dkvmn(skillid, skill_correctness, correctness)

    mask = 1 - tf.cast(tf.equal(correctness, -1), tf.int32)

    mask = tf.squeeze(mask)
    # mask掉
    probility = tf.boolean_mask(probility, mask)
    label = tf.boolean_mask(correctness, mask)

    label = tf.one_hot(label, depth=2)
    # 求auc
    vauc.update_state(label, probility)


def train_one_step(skillid, skill_correctness, correctness):
    with tf.GradientTape() as tape:
        probility = dkvmn(skillid, skill_correctness, correctness)
        mask = 1 - tf.cast(tf.equal(correctness, -1), tf.int32)
        mask = tf.squeeze(mask)
        # mask 掉
        probility = tf.boolean_mask(probility, mask)
        label = tf.boolean_mask(correctness, mask)

        label = tf.one_hot(label, depth=2)
        # 求cc
        bc.update_state(label, probility)

        loss = tf.losses.categorical_crossentropy(label, probility)
        # 求 auc
        auc.update_state(label, probility)

        gradients = tape.gradient(loss, dkvmn.trainable_variables)
        # 反向传播，自动微分计算
        optimizer.apply_gradients(zip(gradients, dkvmn.trainable_variables))

for epoch in range(10):
    train_data = train_data.shuffle(32)
    auc.reset_states()
    vauc.reset_states()
    bc.reset_states()
    for  s, v, l in train_data.as_numpy_iterator():
        train_one_step(s, v, l)

    for s, v, l in val_data.as_numpy_iterator():
        test_one_step(s, v, l)

    with summary_writer.as_default():
        tf.summary.scalar('train_auc', auc.result(), step=epoch)
        tf.summary.scalar('val_auc', vauc.result(), step=epoch)

    print(bc.result(), auc.result(), vauc.result())