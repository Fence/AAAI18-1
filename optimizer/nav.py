import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
from cells.nav import NAVICell
from domains.nav import NAVI_BILINEAR

DEFAULT_SETTINGS = {
    "dims": 2,
    "min_maze_bound": tf.constant(0.0,dtype=tf.float32),
    "max_maze_bound": tf.constant(10.0,dtype=tf.float32),
    "min_act_bound": tf.constant(-1,dtype=tf.float32),
    "max_act_bound": tf.constant(1,dtype=tf.float32),
    "goal": tf.constant(8.0,dtype=tf.float32),
    "centre": tf.constant(5.0,dtype=tf.float32)
   }

class NAVOptimizer(object):
    def __init__(self,
                 a,  # Actions
                 num_step,  # Number of RNN step, this is a fixed step RNN sequence, 12 for navigation
                 num_act,  # Number of actions
                 batch_size,  # Batch Size
                 learning_rate=0.005):
        self.action = tf.reshape(a, [-1, num_step, num_act])  # Reshape rewards
        print(self.action)
        self.batch_size = batch_size
        self.num_step = num_step
        self.learning_rate = learning_rate
        self._p_create_rnn_graph()
        self._p_Q_loss()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _p_create_rnn_graph(self):
        cell = NAVICell(NAVI_BILINEAR, self.batch_size, DEFAULT_SETTINGS)
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        print('action batch size:{0}'.format(array_ops.shape(self.action)[0]))
        print('Initial_state shape:{0}'.format(initial_state))
        rnn_outputs, state = tf.nn.dynamic_rnn(cell, self.action, dtype=tf.float32, initial_state=initial_state)
        # need output intermediate states as well
        concated = tf.concat(axis=0, values=rnn_outputs)
        print('concated shape:{0}'.format(concated.get_shape()))
        something_unpacked = tf.unstack(concated, axis=2)
        self.outputs = tf.reshape(something_unpacked[0], [-1, self.num_step, 1])
        print(' self.outputs:{0}'.format(self.outputs.get_shape()))
        self.intern_states = tf.stack([something_unpacked[1], something_unpacked[2]], axis=2)
        self.last_state = state
        self.pred = tf.reduce_sum(self.outputs, 1)
        self.average_pred = tf.reduce_mean(self.pred)
        print("self.pred:{0}".format(self.pred))

    def _p_create_loss(self):

        objective = tf.reduce_mean(tf.square(self.pred))
        self.loss = objective
        print(self.loss.get_shape())
        # self.loss = -objective
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[a])

    def _p_Q_loss(self):
        objective = tf.constant(0.0, shape=[self.batch_size, 1])
        for i in range(self.num_step):
            Rt = self.outputs[:, i]
            SumRj = tf.constant(0.0, shape=[self.batch_size, 1])
            # SumRk=tf.constant(0.0, shape=[self.batch_size, 1])
            if i < (self.num_step - 1):
                j = i + 1
                SumRj = tf.reduce_sum(self.outputs[:, j:], 1)
                # if i<(self.num_step-1):
                # k= i+1
                # SumRk = tf.reduce_sum(self.outputs[:,k:],1)
            objective += (Rt * SumRj + tf.square(Rt)) * (self.num_step - i) / np.square(self.num_step)
        self.loss = tf.reduce_mean(objective)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[a])

    def Optimize(self, epoch=100, show_progress=False):
        new_loss = self.sess.run([self.loss])
        print('Loss in epoch {0}: {1}'.format("Initial", new_loss))
        if show_progress:
            progress = []
        for epoch in range(epoch):
            training = self.sess.run([self.optimizer])
            self.sess.run(
                tf.assign(a, tf.clip_by_value(a, DEFAULT_SETTINGS['min_act_bound'], DEFAULT_SETTINGS['max_act_bound'])))
            if True:
                new_loss = self.sess.run([self.average_pred])
                print('Loss in epoch {0}: {1}'.format(epoch, new_loss))
            if show_progress and epoch % 10 == 0:
                progress.append(self.sess.run(self.intern_states))
        minimum_costs_id = self.sess.run(tf.argmax(self.pred, 0))
        print(minimum_costs_id)
        best_action = np.round(self.sess.run(self.action)[minimum_costs_id[0]], 4)
        print('Optimal Action Squence:{0}'.format(best_action))
        pred_list = self.sess.run(self.pred)
        pred_list = np.sort(pred_list.flatten())[::-1]
        pred_list = pred_list[:5]
        pred_mean = np.mean(pred_list)
        pred_std = np.std(pred_list)
        print('Best Cost: {0}'.format(pred_list[0]))
        print('Sorted Costs:{0}'.format(pred_list))
        print('MEAN: {0}, STD:{1}'.format(pred_mean, pred_std))
        print('The last state:{0}'.format(self.sess.run(self.last_state)[minimum_costs_id[0]]))
        print('Rewards each time step:{0}'.format(self.sess.run(self.outputs)[minimum_costs_id[0]]))
        print('Intermediate states:{0}'.format(self.sess.run(self.intern_states)[minimum_costs_id[0]]))
        if show_progress:
            progress = np.array(progress)[:, minimum_costs_id[0]]
            print('progress shape:{0}'.format(progress.shape))
            np.savetxt("progress.csv", progress.reshape((progress.shape[0], -1)), delimiter=",", fmt='%2.5f')
