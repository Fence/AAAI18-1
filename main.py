import tensorflow as tf
from optimizer.hvac import HVACOptimizer
from optimizer.nav import NAVOptimizer
from optimizer.reservoir import ReservoirOptimizer
from domains.nav import NAVI_NONLINEAR, NAVI_BILINEAR, NAVI_LINEAR
from domains.reservoir import RESERVOIR_LINEAR, RESERVOIR_NONLINEAR
from instances.hvac import HVAC_60

# sess = tf.Session()
# initial_a = tf.truncated_normal(shape=[100, 96, 60], mean=5.0, stddev=1.0).eval(session=sess)
# a = tf.Variable(initial_a, name="action")
# rnn_inst = HVACOptimizer(a, HVAC_60, 96, 100)
# rnn_inst.Optimize(4000)

# sess = tf.Session()
# initial_a = tf.truncated_normal(shape=[100, 120, 2], mean=0.0, stddev=0.005).eval(session=sess)
# a = tf.Variable(initial_a, name="action")
# rnn_inst = NAVOptimizer(a, 120, 100, NAVI_NONLINEAR)
# rnn_inst.Optimize(300)


sess = tf.Session()
initial_a = tf.truncated_normal(shape=[100, 120, 20], mean=0.0, stddev=0.5).eval(session=sess)
a = tf.Variable(initial_a, name="action")
rnn_inst = ReservoirOptimizer(a, 120, 100, RESERVOIR_NONLINEAR)
rnn_inst.Optimize(300)