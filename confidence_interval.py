import os
import gc
import json
import numpy as np
import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiments.configuration import CONFIGURATIONS


def multiple_run(config, num_run=10):
    for index, step in enumerate(config['step']):
        name = (config['optimizer'].__name__
                + ' ' + config['domain'].__name__
                + ' ' + config['instance'][index]['name']
                + ' Planning Step:' + str(step))
        print name
        records = []
        for i in xrange(num_run):
            gc.collect()
            tf.reset_default_graph()
            with tf.Session() as sess:
                initial_a = tf.truncated_normal(shape=[config['batch'], step, config['dimension']],
                                                mean=config['initial_mean'],
                                                stddev=config['initial_std'])
                a = tf.Variable(initial_a, name="action")
                rnn_inst = config['optimizer'](a, step, config['batch'], config['domain'], config['instance'][index],
                                               sess=sess)
                the_best = rnn_inst.Optimize(config['epoch'], showbest=True)
                records.append(the_best)
        mean = np.mean(records)
        std = np.std(records)
        ci = 1.96*std/np.sqrt(10)
        print "mean:{0}, std:{1}".format(mean, ci)

for config in CONFIGURATIONS:
    multiple_run(config, 10)