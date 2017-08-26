import tensorflow as tf
from domains.hvac import HVAC


class HVACCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, adj_outside, adj_hall, adj, rooms, batch_size, default_settings):
        self._num_state_units = len(rooms)
        self._num_reward_units = 1+len(rooms)
        self.hvac = HVAC(adj_outside, adj_hall, adj, rooms, batch_size, default_settings)

    @property
    def state_size(self):
        return self._num_state_units

    @property
    def output_size(self):
        return self._num_reward_units

    def __call__(self, inputs, state, scope=None):
        next_state = self.hvac.Transition(state, inputs)
        reward = self.hvac.Reward(state, inputs)
        return tf.concat(axis=1, values=[reward, next_state]), next_state
