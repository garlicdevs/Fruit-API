class MapBaseNetwork(object):
    def __init__(self, network_config,
                 using_gpu=True,
                 load_model_path=None,
                 num_of_checkpoints=5):

        if network_config is None:
            raise ValueError("Network requires a configuration to work!!")

        self.network_config = network_config
        self.using_gpu = using_gpu
        self.stochastic_policy = network_config.is_stochastic()
        self.load_model_path = load_model_path
        self.save = True

        with tf.device('/gpu:0' if self.using_gpu else '/cpu:0'):
            with tf.Graph().as_default() as graph:
                self.tf_graph = graph

                self.tf_network_variables = self.create_network()

                self.tf_saver = tf.train.Saver(var_list=self.tf_network_variables, max_to_keep=num_of_checkpoints)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.tf_session = tf.Session(config=config)

                if self.load_model_path is None:
                    self.tf_session.run(tf.global_variables_initializer())
                else:
                    self.tf_session.run(tf.global_variables_initializer())
                    self.load_model(self.load_model_path)

    def create_network(self):
        return self.network_config.init_config()

    def save_model(self, *args, **kwargs):
        if self.save:
            self.tf_saver.save(self.tf_session, *args, **kwargs)

    def load_model(self, path):
        self.tf_saver.restore(self.tf_session, path)

    def get_action(self, probs):
        if self.stochastic_policy:
            action_probs = probs - np.finfo(np.float32).epsneg
            sample = np.random.multinomial(1, action_probs)
            action_index = int(np.nonzero(sample)[0])
            return action_index
        else:
            return np.argmax(probs)

    def get_output(self, state, map):
        pass

    def train_network(self, state, action, reward, state_tp1s, terminal, maps, learning_rate, global_step=None,
                      summaries=False, other=None):
        pass

    def reset_network(self):
        self.network_config.reset_config()

    def get_epsilon_greedy_action(self, state, epsilon):
        probs = self.network_config.get_prediction(self.tf_session, state)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, len(probs))
        else:
            return np.argmax(probs)

    def get_graph(self):
        return self.tf_graph

    def get_session(self):
        return self.tf_session

    def get_config(self):
        return self.network_config

    def set_save_model(self, save):
        self.save = save


class NIPSBaseNetwork(object):
    def __init__(self, network_config,
                 using_gpu=True,
                 load_model_path=None,
                 num_of_checkpoints=5):

        if network_config is None:
            raise ValueError("Network requires a configuration to work!!")

        self.network_config = network_config
        self.using_gpu = using_gpu
        self.stochastic_policy = network_config.is_stochastic()
        self.load_model_path = load_model_path
        self.save = True

        with tf.device('/gpu:0' if self.using_gpu else '/cpu:0'):
            with tf.Graph().as_default() as graph:
                self.tf_graph = graph

                self.tf_network_variables = self.create_network()

                self.tf_saver = tf.train.Saver(var_list=self.tf_network_variables, max_to_keep=num_of_checkpoints)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.tf_session = tf.Session(config=config)

                if self.load_model_path is None:
                    self.tf_session.run(tf.global_variables_initializer())
                else:
                    self.tf_session.run(tf.global_variables_initializer())
                    self.load_model(self.load_model_path)

    def create_network(self):
        return self.network_config.init_config()

    def save_model(self, *args, **kwargs):
        if self.save:
            self.tf_saver.save(self.tf_session, *args, **kwargs)

    def load_model(self, path):
        self.tf_saver.restore(self.tf_session, path)

    def get_action(self, probs):
        if self.stochastic_policy:
            action_probs = probs - np.finfo(np.float32).epsneg
            sample = np.random.multinomial(1, action_probs)
            action_index = int(np.nonzero(sample)[0])
            return action_index
        else:
            return np.argmax(probs)

    def get_output(self, x, y):
        pass

    def train_network(self, state, action, reward, state_tp1s, terminal, map_data, learning_rate, global_step=None,
                      summaries=False, other=None):
        pass

    def reset_network(self):
        self.network_config.reset_config()

    def get_epsilon_greedy_action(self, state, epsilon):
        probs = self.network_config.get_prediction(self.tf_session, state)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, len(probs))
        else:
            return np.argmax(probs)

    def get_graph(self):
        return self.tf_graph

    def get_session(self):
        return self.tf_session

    def get_config(self):
        return self.network_config

    def set_save_model(self, save):
        self.save = save


class MANIPSBaseNetwork(object):
    def __init__(self, network_config,
                 using_gpu=True,
                 load_model_path=None,
                 load_model_path_2=None,
                 num_of_checkpoints=5):

        if network_config is None:
            raise ValueError("Network requires a configuration to work!!")

        self.network_config = network_config
        self.using_gpu = using_gpu
        self.stochastic_policy = network_config.is_stochastic()
        self.load_model_path = load_model_path
        self.load_model_path_2 = load_model_path_2
        self.save = True

        with tf.device('/gpu:0' if self.using_gpu else '/cpu:0'):
            with tf.Graph().as_default() as graph:
                self.tf_graph = graph

                self.tf_network_variables = self.create_network()

                self.tf_saver = tf.train.Saver(var_list=self.tf_network_variables, max_to_keep=num_of_checkpoints)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.tf_session = tf.Session(config=config)

                if self.load_model_path is None:
                    self.tf_session.run(tf.global_variables_initializer())
                else:
                    self.tf_session.run(tf.global_variables_initializer())
                    self.load_model_2(self.load_model_path_2)
                    self.load_model(self.load_model_path)

    def create_network(self):
        return self.network_config.init_config()

    def save_model(self, *args, **kwargs):
        if self.save:
            self.tf_saver.save(self.tf_session, *args, **kwargs)

    def load_model(self, path):
        self.tf_saver.restore(self.tf_session, path)

    def load_model_2(self, path):
        self.tf_saver.restore(self.tf_session, path)
        self.network_config.update_network_2(self.tf_session)

    def get_action(self, probs):
        if self.stochastic_policy:
            action_probs = probs - np.finfo(np.float32).epsneg
            sample = np.random.multinomial(1, action_probs)
            action_index = int(np.nonzero(sample)[0])
            return action_index
        else:
            return np.argmax(probs)

    def get_output(self, x, y, z):
        pass

    def train_network(self, state, action, reward, state_tp1s, terminal, map_data, learning_rate, global_step=None,
                      summaries=False, other=None):
        pass

    def reset_network(self):
        self.network_config.reset_config()

    def get_epsilon_greedy_action(self, state, epsilon):
        probs = self.network_config.get_prediction(self.tf_session, state)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, len(probs))
        else:
            return np.argmax(probs)

    def get_graph(self):
        return self.tf_graph

    def get_session(self):
        return self.tf_session

    def get_config(self):
        return self.network_config

    def set_save_model(self, save):
        self.save = save


class DQBaseNetwork(object):
    def __init__(self, network_config,
                 using_gpu=True,
                 load_model_path=None,
                 load_model_path_2=None,
                 num_of_checkpoints=5):

        if network_config is None:
            raise ValueError("Network requires a configuration to work!!")

        self.network_config = network_config
        self.using_gpu = using_gpu
        self.stochastic_policy = network_config.is_stochastic()
        self.load_model_path = load_model_path
        self.load_model_path_2 = load_model_path_2
        self.save = True
        self.num_of_checkpoints = num_of_checkpoints

        with tf.device('/gpu:0' if self.using_gpu else '/cpu:0'):
            with tf.Graph().as_default() as graph:
                self.tf_graph = graph

                self.tf_network_variables, self.tf_network_variables_2 = self.create_network()

                self.tf_saver = tf.train.Saver(var_list=self.tf_network_variables, max_to_keep=num_of_checkpoints)

                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.tf_session = tf.Session(config=config)

                if self.load_model_path is None:
                    self.tf_session.run(tf.global_variables_initializer())
                else:
                    self.tf_session.run(tf.global_variables_initializer())
                    self.load_model(self.load_model_path, self.load_model_path_2)

    def create_network(self):
        return self.network_config.init_config()

    def save_model(self, *args, **kwargs):
        if self.save:
            self.tf_saver.save(self.tf_session, *args, **kwargs)

    def load_model(self, path, path_2):
        self.tf_saver.restore(self.tf_session, path_2)
        self.network_config.update_dq_network(self.tf_session)
        self.tf_saver = tf.train.Saver(var_list=self.tf_network_variables, max_to_keep=self.num_of_checkpoints)
        self.tf_saver.restore(self.tf_session, path)

    def get_action(self, probs_1, probs_2, alpha, epsilon):
        gen = np.random.uniform(0,1)
        if gen < epsilon:
            return np.random.randint(0, self.network_config.get_output_size())
        else:
            gen = np.random.uniform(0, 1)
            if gen < alpha:
                action_probs = probs_1 - np.finfo(np.float32).epsneg
                sample = np.random.multinomial(1, action_probs)
                action_index = int(np.nonzero(sample)[0])
                return action_index
            else:
                action_probs = probs_2 - np.finfo(np.float32).epsneg
                sample = np.random.multinomial(1, action_probs)
                action_index = int(np.nonzero(sample)[0])
                return action_index

    def get_output(self, x):
        pass

    def train_network(self, state, action, reward, state_tp1s, terminal, learning_rate, global_step=None,
                      summaries=False, other=None):
        pass

    def reset_network(self):
        self.network_config.reset_config()

    def get_epsilon_greedy_action(self, state, epsilon):
        probs = self.network_config.get_prediction(self.tf_session, state)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, len(probs))
        else:
            return np.argmax(probs)

    def get_graph(self):
        return self.tf_graph

    def get_session(self):
        return self.tf_session

    def get_config(self):
        return self.network_config

    def set_save_model(self, save):
        self.save = save