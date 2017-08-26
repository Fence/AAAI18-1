import tensorflow as tf
from optimizer.hvac import ActionOptimizer
from instances.hvac import HVAC_60

sess = tf.InteractiveSession()
initial_a = tf.truncated_normal(shape=[100, 96, 60], mean=5.0, stddev=1.0).eval()
a = tf.Variable(initial_a, name="action")
rnn_inst = ActionOptimizer(a, HVAC_60, 96, 100, "MSE")
rnn_inst.Optimize(4000)
