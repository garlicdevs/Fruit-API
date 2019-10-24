class JairPolicyNetwork(BaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=num_of_checkpoints
                         )

    def get_output_with_weight(self, state, thread_id):
        probs = self.network_config.get_prediction_with_weight(self.tf_session, state, thread_id)
        return self.get_action(probs=probs)

    def train_network(self, states, acts, rewards, states_target, terminals,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 current_learning_rate, summaries, global_step, other)


# class MAPolicyNetwork(BaseNetwork):
#     def __init__(self, network_config, using_gpu=True, load_model_path=None, num_of_checkpoints=5):
#
#         super().__init__(
#                          network_config=network_config,
#                          using_gpu=using_gpu,
#                          load_model_path=load_model_path,
#                          num_of_checkpoints=num_of_checkpoints
#                          )
#
#     def get_output(self, state):
#         probs = self.network_config.get_prediction(self.tf_session, state)
#         probs_2 = self.network_config.get_prediction_2(self.tf_session, state)
#         return [self.get_action(probs=probs), self.get_action(probs=probs_2)]
#
#     def train_network(self, states, acts, rewards, states_target, terminals,
#                       current_learning_rate, global_step=0, summaries=False, other=None):
#
#         return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
#                                                  current_learning_rate, summaries, global_step, other)

class MAPolicyNetwork(BaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=num_of_checkpoints
                         )

    def get_output(self, state):
        agents = self.network_config.env.get_num_of_agents()

        probs = self.network_config.get_prediction(self.tf_session, state)

        actions = []
        for i in range(agents):
            actions.append(self.get_action(probs[i][0]))

        return actions

    def train_network(self, states, acts, rewards, states_target, terminals,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 current_learning_rate, summaries, global_step, other)


class MapMAPolicyNetwork(MapBaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=num_of_checkpoints
                         )

    def get_output(self, state, map):
        agents = self.network_config.env.get_num_of_agents()

        probs = self.network_config.get_prediction(self.tf_session, state, map)

        actions = []
        for i in range(agents):
            actions.append(self.get_action(probs[i][0]))

        return actions

    def train_network(self, states, acts, rewards, states_target, terminals, maps,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 maps, current_learning_rate, summaries, global_step, other)


class NIPSPolicyNetwork(NIPSBaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=num_of_checkpoints
                         )

    def get_output(self, state, map_data):
        probs = self.network_config.get_prediction_with_map(self.tf_session, state, map_data)
        return self.get_action(probs=probs)

    def train_network(self, states, acts, rewards, states_target, terminals, map_data,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 current_learning_rate, summaries, global_step, map_data)


class MANIPSPolicyNetwork(MANIPSBaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, load_model_path_2=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         load_model_path_2=load_model_path_2,
                         num_of_checkpoints=num_of_checkpoints
                         )

    def get_output(self, state, map_data, map_data_2):
        probs = self.network_config.get_prediction_with_map(self.tf_session, state, map_data)
        act_1 = self.get_action(probs=probs)

        probs_2 = self.network_config.get_prediction_with_map_2(self.tf_session, state, map_data_2)
        act_2 = self.get_action(probs=probs_2)

        return [act_1, act_2]

    def train_network(self, states, acts, rewards, states_target, terminals, map_data,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 current_learning_rate, summaries, global_step, map_data)


class MSCSPolicyNetwork(NIPSBaseNetwork):
    def __init__(self, network_config, network_1, network_2, using_gpu=True, load_model_path=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=num_of_checkpoints
                         )

        self.network_1 = network_1
        self.network_2 = network_2

        self.network_changed = False
        self.current_network = 1
        self.regional_steps = 3

    def get_output(self, state, map_data):
        #probs = self.network_config.get_prediction_with_map(self.tf_session, state, map_data)
        #index = self.get_action(probs=probs)
        index = self.network_config.env.get_key_pressed()
        if self.current_network == 1 and index == 257:
            self.network_changed = True
            self.regional_steps = 3
        elif self.current_network == 2 and index != 257:
            self.network_changed = True
            self.regional_steps = 3
        else:
            self.network_changed = False
        if index == 257:
            print("Network 2")
            self.current_network = 2
            return self.network_2.get_output(state, map_data)
        else:
            print("Network 1")
            self.current_network = 1
            if self.regional_steps > 0:
                self.regional_steps = self.regional_steps - 1
                return 2
            return self.network_1.get_output(state, map_data)

    def train_network(self, states, acts, rewards, states_target, terminals, map_data,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 current_learning_rate, summaries, global_step, map_data)


class DQPolicyNetwork(DQBaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, load_model_path_2=None, num_of_checkpoints=5,
                 alpha=0.2, epsilon=0.02):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         load_model_path_2=load_model_path_2,
                         num_of_checkpoints=num_of_checkpoints
                         )
        self.alpha = alpha
        self.epsilon = epsilon

    def get_output(self, state):
        probs_1, probs_2 = self.network_config.get_prediction(self.tf_session, state)
        return self.get_action(probs_1=probs_1, probs_2=probs_2, alpha=self.alpha, epsilon=self.epsilon)

    def train_network(self, states, acts, rewards, states_target, terminals,
                      current_learning_rate, global_step=0, summaries=False, other=None):

        return self.network_config.train_network(self.tf_session, states, acts, rewards, states_target, terminals,
                                                 current_learning_rate, summaries, global_step, other)


class MOPolicyNetwork(PolicyNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, num_of_checkpoints=5):

        super().__init__(
                         network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=num_of_checkpoints
                         )

        self.num_of_objs = network_config.get_num_of_objectives()
        self.num_of_actions = network_config.get_output_size()