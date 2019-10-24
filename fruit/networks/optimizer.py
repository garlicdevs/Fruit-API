from functools import partial

import tensorflow as tf


class OptimizerFactory():
    @staticmethod
    def get_optimizer(method='RMSProp-1'):
        if method == 'RMSProp-1':
            return partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)
        elif method == 'RMSProp-2':
            return partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=1e-6)
        elif method == 'Adam':
            return partial(tf.train.AdamOptimizer)
        else:
            raise ValueError('Method {} is not supported yet !'.format(method))