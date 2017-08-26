import tensorflow as tf

class NAVI_BILINEAR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.batch_size = batch_size
        self.zero = tf.constant(0, shape=[batch_size, 2], dtype=tf.float32)
        self.four = tf.constant(2.0, dtype=tf.float32)
        self.one = tf.constant(1.0, shape=[batch_size], dtype=tf.float32)

    def MINMAZEBOUND(self):
        return self.min_maze_bound

    def MAXMAZEBOUND(self):
        return self.max_maze_bound

    def MINACTIONBOUND(self):
        return self.min_act_bound

    def MAXACTIONBOUND(self):
        return self.max_act_bound

    def GOAL(self):
        return self.goal

    def CENTER(self):
        return self.centre

    def Transition(self, states, actions):
        previous_state = states
        distance = tf.reduce_sum(tf.abs(states - self.CENTER()), 1)
        scalefactor = tf.where(tf.less(distance, self.four), distance / self.four, self.one)
        proposedLoc = previous_state + tf.matrix_transpose(scalefactor * tf.matrix_transpose(actions))
        new_states = tf.where(tf.logical_and(tf.less_equal(proposedLoc, self.MAXMAZEBOUND()),
                                             tf.greater_equal(proposedLoc, self.MINMAZEBOUND())),
                              proposedLoc,
                              tf.where(tf.greater(proposedLoc, self.MAXMAZEBOUND()),
                                       self.zero + self.MAXMAZEBOUND(),
                                       self.zero + self.MINMAZEBOUND())
                              )
        return new_states

    def Reward(self, states, actions):
        new_reward = -tf.reduce_sum(tf.abs(states - self.GOAL()), 1, keep_dims=True)
        return new_reward