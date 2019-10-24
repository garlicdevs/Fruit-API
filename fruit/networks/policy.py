from fruit.networks.base import BaseNetwork


class PolicyNetwork(BaseNetwork):
    def __init__(self, network_config, using_gpu=True, load_model_path=None, max_num_of_checkpoints=50):

        super().__init__(network_config=network_config,
                         using_gpu=using_gpu,
                         load_model_path=load_model_path,
                         num_of_checkpoints=max_num_of_checkpoints
                         )

    def predict(self, state):
        probs = self.network_config.predict(self.tf_session, state)
        return probs

    def train_network(self, data_dict):
        return self.network_config.train(self.tf_session, data_dict)



