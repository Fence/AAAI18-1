import os
import gc
import json
import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiments.configuration import CONFIGURATIONS

Data = {}
for config in CONFIGURATIONS:
    for index, step in enumerate(config['step']):
        gc.collect()
        tf.reset_default_graph()
        with tf.Session() as sess:
            name = (config['optimizer'].__name__
                    + ' ' + config['domain'].__name__
                    + ' ' + config['instance'][index]['name']
                    + ' Planning Step:' + str(step))
            print name
            initial_a = tf.truncated_normal(shape=[config['batch'], step, config['dimension']],
                                            mean=config['initial_mean'],
                                            stddev=config['initial_std'])
            a = tf.Variable(initial_a, name="action")
            rnn_inst = config['optimizer'](a, step, config['batch'], config['domain'], config['instance'][index], sess=sess)
            mean, std = rnn_inst.Optimize(config['epoch'])
            print 'mean: {0}, std: {1}'.format(mean, std)
            Data[name] = unicode([mean, std])
with open('result.json', 'w') as fp:
    json.dump(Data, fp)