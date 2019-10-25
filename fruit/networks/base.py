import tensorflow as tf


class BaseNetwork(object):
    def __init__(self, network_config,
                 using_gpu=True,
                 load_model_path=None,
                 num_of_checkpoints=50):

        if network_config is None:
            raise ValueError("Network requires a configuration to work!!")

        self.network_config = network_config
        self.using_gpu = using_gpu
        self.load_model_path = load_model_path
        self.save = True

        with tf.device('/gpu:0' if self.using_gpu else '/cpu:0'):
            with tf.Graph().as_default() as graph:
                self.tf_graph = graph

                self.tf_network_variables = self.create_network()

                if num_of_checkpoints < 0:
                    num_of_checkpoints = 1000
                self.num_of_checkpoints = num_of_checkpoints
                self.tf_saver = tf.train.Saver(var_list=self.tf_network_variables, max_to_keep=num_of_checkpoints)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.tf_session = tf.Session(config=config)

                if self.load_model_path is None:
                    self.tf_session.run(tf.global_variables_initializer())
                else:
                    self.tf_session.run(tf.global_variables_initializer())
                    self.load_model()

    def create_network(self):
        return self.network_config.init_config()

    def save_model(self, *args, **kwargs):
        if self.save:
            self.tf_saver.save(self.tf_session, *args, **kwargs)

    def load_model(self, path=None):
        if path is None:
            self.tf_saver.restore(self.tf_session, self.load_model_path)
        else:
            self.tf_saver.restore(self.tf_session, path)

    def predict(self, x):
        pass

    def train_network(self, data_dict):
        pass

    def reset_network(self):
        self.network_config.reset_config()

    def get_graph(self):
        return self.tf_graph

    def get_session(self):
        return self.tf_session

    def get_config(self):
        return self.network_config

    def set_save_model(self, save_model):
        self.save = save_model

