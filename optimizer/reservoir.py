import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
from cells.reservoir import RESERVOIRCell


DEFAULT_SETTINGS = {
    "max_cap": [100,100,100,100,100,100,100,100,200,200,200,200,200,200,400,400,400,500,500,1000],
    "high_bound": [80,80,80,80,80,80,80,80,180,180,180,180,180,180,380,380,380,480,480,980],
    "low_bound": [20,20,20,20,20,20,20,20,30,30,30,30,30,30,40,40,40,60,60,100],
    "rain": [5,5,5,5,5,5,5,5,10,10,10,10,10,10,20,20,20,30,30,40],
    "downstream": [[1,9],[2,9],[3,10],[4,10],[5,11],[6,11],[7,12],[8,12],[9,15],[10,15],
                    [11,16],[12,16],[13,17],[14,17],[15,18],[16,19],[17,19],[18,20],[19,20]],
    "downtosea": [20],
    "biggestmaxcap": 1000,
    "reservoirs": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "init_state": [75,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50]
}


class ActionOptimizer(object):
    def __init__(self,
                 action,  # Actions
                 num_step,  # Number of RNN step, this is a fixed step RNN sequence, 12 for navigation
                 num_act,  # Number of actions
                 batch_size,  # Batch Size
                 learning_rate=0.1):
        self.action = action,
        print(self.action)
        self.batch_size = batch_size
        self.num_step = num_step
        self.learning_rate = learning_rate
        self._p_create_rnn_graph()
        self._p_create_loss()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _p_create_rnn_graph(self):
        cell = RESERVOIRCell(self.batch_size, DEFAULT_SETTINGS)
        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32) + tf.constant(
            [DEFAULT_SETTINGS["init_state"]], dtype=tf.float32)
        print('action batch size:{0}'.format(array_ops.shape(self.action)[0]))
        print('Initial_state shape:{0}'.format(initial_state))
        rnn_outputs, state = tf.nn.dynamic_rnn(cell, self.action, dtype=tf.float32, initial_state=initial_state)
        # need output intermediate states as well
        concated = tf.concat(0, rnn_outputs)
        print('concated shape:{0}'.format(concated.get_shape()))
        something_unpacked = tf.unstack(concated, axis=2)
        self.outputs = tf.reshape(something_unpacked[0], [-1, self.num_step, 1])
        print(' self.outputs:{0}'.format(self.outputs.get_shape()))
        self.intern_states = tf.stack([something_unpacked[i + 1] for i in range(len(DEFAULT_SETTINGS["reservoirs"]))],
                                     axis=2)
        self.last_state = state
        self.pred = tf.reduce_sum(self.outputs, 1)
        self.average_pred = tf.reduce_mean(self.pred)
        print("self.pred:{0}".format(self.pred))

    def _p_create_loss(self):
        print("MSE-loss")
        objective = tf.reduce_mean(tf.square(self.pred))
        self.loss = objective
        print(self.loss.get_shape())
        # self.loss = -objective
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.action])

    def Optimize(self, epoch=100):
        #         Time_Target_List = [15,30,60,120,240,480,960]
        #         Target = Time_Target_List[0]
        #         counter = 0
        #         new_loss = self.sess.run([self.average_pred])
        #         print('Loss in epoch {0}: {1}'.format("Initial", new_loss))
        #         print('Compile to backend complete!')
        #         start = time.time()
        #         while True:
        #             training = self.sess.run([self.optimizer])
        #             action_upperbound=self.sess.run(self.intern_states)
        #             self.sess.run(tf.assign(self.action, tf.clip_by_value(self.action, 0, action_upperbound)))
        #             end = time.time()
        #             if end-start>=Target:
        #                 print('Time: {0}'.format(Target))
        #                 pred_list = self.sess.run(self.pred)
        #                 pred_list=np.sort(pred_list.flatten())[::-1]
        #                 pred_list=pred_list[:5]
        #                 pred_mean = np.mean(pred_list)
        #                 pred_std = np.std(pred_list)
        #                 print('Best Cost: {0}'.format(pred_list[0]))
        #                 print('MEAN: {0}, STD:{1}'.format(pred_mean,pred_std))
        #                 counter = counter+1
        #                 if counter == len(Time_Target_List):
        #                     print("Done!")
        #                     break
        #                 else:
        #                     Target = Time_Target_List[counter]

        new_loss = self.sess.run([self.average_pred])
        print('Loss in epoch {0}: {1}'.format("Initial", new_loss))
        for epoch in range(epoch):
            training = self.sess.run([self.optimizer])
            action_upperbound = self.sess.run(self.intern_states)
            self.sess.run(tf.assign(self.action, tf.clip_by_value(self.action, 0, action_upperbound)))
            if True:
                new_loss = self.sess.run([self.average_pred])
                print('Loss in epoch {0}: {1}'.format(epoch, new_loss))
        minimum_costs_id = self.sess.run(tf.argmax(self.pred, 0))
        print(minimum_costs_id)
        best_action = self.sess.run(self.action)[minimum_costs_id[0]]
        np.savetxt("RS_ACTION.csv", best_action, delimiter=",", fmt='%2.5f')
        print('Optimal Action Squence:{0}'.format(best_action))
        pred_list = self.sess.run(self.pred)
        pred_list = np.sort(pred_list.flatten())[::-1]
        pred_list = pred_list[:5]
        pred_mean = np.mean(pred_list)
        pred_std = np.std(pred_list)
        print('Best Cost: {0}'.format(pred_list[0]))
        print('Sorted Costs:{0}'.format(pred_list))
        print('MEAN: {0}, STD:{1}'.format(pred_mean, pred_std))
        print('The last state:{0}'.format(self.sess.run(self.last_state)))
        print('The last state:{0}'.format(self.sess.run(self.last_state)[minimum_costs_id[0]]))
        print('Rewards each time step:{0}'.format(self.sess.run(self.outputs)[minimum_costs_id[0]]))
        print('Intermediate states:{0}'.format(self.sess.run(self.intern_states)[minimum_costs_id[0]]))
        interm = self.sess.run(self.intern_states)[minimum_costs_id[0]]
        np.savetxt("RS_INTERM.csv", interm, delimiter=",", fmt='%2.5f')

