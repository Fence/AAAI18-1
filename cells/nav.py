import tensorflow as tf
from domains.nav_bilinear import NAVI_BILINEAR

class NAVICell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, batch_size, default_settings):
        self._num_state_units = 2
        self._num_reward_units = 3
        self.navi = NAVI_BILINEAR(batch_size, default_settings)

    @property
    def state_size(self):
        return self._num_state_units

    @property
    def output_size(self):
        return self._num_reward_units

    def __call__(self, inputs, state, scope=None):
        next_state =  self.navi.Transition(state, inputs)
        reward = self.navi.Reward(state, inputs)
        return tf.concat(1, [reward, next_state]), next_state

