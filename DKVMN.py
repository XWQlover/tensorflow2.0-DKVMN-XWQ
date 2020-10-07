import tensorflow as tf
total_skills_correctness = 200
total_skills = 100
embedding_size = 100
batchsize = 32
M = 50


class DKVMNcell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, **kwargs):
        self.units = units
        super(DKVMNcell, self).__init__(**kwargs)
        self.Mv = self.add_weight(shape=(M, embedding_size),
                                  initializer='random_normal',
                                  trainable=True)
        self.Mv = tf.expand_dims(self.Mv, axis=0)

    @property
    def state_size(self):
        return self.units

    def call(self, w_attention, erase_signal_mul, add_signal_mul, states):
        """
        :param w_attention: 这个应该是concept矩阵计算后的注意力权重
        :param erase_signal: erase标志
        :param add_signal: add标志
        :param states: Mk矩阵
        :return: r，Mv
        """
        # 读
        # w_attention.shape (50,50) state.shape (50,50,100)
        r = tf.matmul(tf.expand_dims(w_attention, axis=1), states)

        # print(r.shape)(50,1,100)
        r = r[:, 0, :]

        # 写
        states = states * erase_signal_mul + add_signal_mul

        return r, states


class DKVMN(tf.keras.models.Model):
    def __init__(self):
        super(DKVMN, self).__init__()
        # 掩码层
        self.mask = tf.keras.layers.Masking(mask_value=-1)
        # 题目嵌入
        self.exercise_embedding = tf.keras.layers.Embedding(total_skills, embedding_size)
        # 题目对错嵌入
        self.exercise_correctness_embedding = tf.keras.layers.Embedding(total_skills_correctness, embedding_size)

        self.cell = DKVMNcell(10)

        self.Mk = self.add_weight(shape=(M, embedding_size),
                                  initializer='random_normal',
                                  trainable=True)

        self.erase = tf.keras.layers.Dense(embedding_size)
        self.add = tf.keras.layers.Dense(embedding_size, activation="tanh")
        self.r = tf.keras.layers.Dense(embedding_size, activation="tanh")
        self.p = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, skillid, skill_correctness, correctness):
        shape = skillid.shape
        skill_correctness = tf.expand_dims(skill_correctness, axis=-1)
        skillid = tf.expand_dims(skillid, axis=-1)
        # 掩码
        skillid = self.mask(skillid)
        skill_correctness = self.mask(skill_correctness)
        # 映射
        skill_embedding = self.exercise_embedding(skillid)
        skill_correctness_embedding = self.exercise_correctness_embedding(
            skill_correctness)  # （batch,seqlen,embeddingsize）
        skill_correctness_embedding = tf.squeeze(skill_correctness_embedding, axis=2)
        skill_embedding = tf.squeeze(skill_embedding, axis=2)
        # tensorlist_batch = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        # for k in range(skill_correctness_embedding.shape[0]):
        #   tensorlist = tf.TensorArray(tf.float32,size=0,dynamic_size=True)
        #   for i in range(skill_correctness_embedding.shape[1]):
        #     # 做对拼接1 ，做错拼接0
        #     w=tensorlist.write(i,tf.cond(correctness[k,i],lambda:tf.zeros_like(skill_correctness_embedding[k,i]),lambda:tf.ones_like(skill_correctness_embedding[k,i])))
        #     w.mark_used()
        #   w=tensorlist_batch.write(k,tensorlist.stack())
        #   w.mark_used()
        # # 拼接后skill_correctness_embedding
        # skill_correctness_embedding = tf.concat([skill_correctness_embedding,tensorlist_batch.stack()],axis=-1)

        # 产生 注意力权重
        w_attention = tf.matmul(skill_embedding, tf.expand_dims(tf.transpose(self.Mk), axis=0))
        w_attention = tf.nn.softmax(w_attention)

        #  遗忘 和 更新 Mv的过程
        erase_signal = self.erase(skill_correctness_embedding)
        add_signal = self.add(skill_correctness_embedding)

        erase_signal_mul = 1 - tf.expand_dims(w_attention, axis=-1) * tf.expand_dims(erase_signal, axis=2)
        add_signal_mul = tf.expand_dims(w_attention, axis=-1) * tf.expand_dims(add_signal, axis=2)
        # 遗忘和更新Mv
        # batch个 Mv
        states = self.cell.Mv
        for i in range(batchsize)[1:]:
            states = tf.concat([states, self.cell.Mv], axis=0)

        cell_out_list = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.float32)

        for i in range(shape[1]):
            r, states = self.cell(w_attention[:, i], erase_signal_mul[:, i], add_signal_mul[:, i], states)
            w = cell_out_list.write(i, tf.expand_dims(r, axis=1))
            w.mark_used()
        f = cell_out_list.read(0)
        for i in range(shape[1])[1:]:
            f = tf.concat([f, cell_out_list.read(i)], axis=1)

        r = tf.concat([f, skill_embedding], axis=-1)
        loss = tf.nn.softmax(self.p(self.r(r)))
        return loss
