from fruit.agents.base import BaseAgent


class AgentFactory(object):
    """
    As its name, the class is used to instantiate the ``BaseAgent`` and a set of user-defined learners.

    """
    @staticmethod
    def get_base_agent(network, environment, num_of_threads=0, checkpoint_frequency=5e4, learner_report_frequency=10,
                       num_of_epochs=50, steps_per_epoch=1e6, log_dir="./train/a3c"):
        """

        :param network: a reference to ``PolicyNetwork``
        :param environment: a reference to the environment
        :param num_of_threads: the number of learners
        :param checkpoint_frequency: checkpoints will be saved with ``checkpoint_frequency``
        :param learner_report_frequency: each learner will print a debug message with ``learner_report_frequency``
        :param num_of_epochs: the number of training epochs.
        :param steps_per_epoch: the number of training steps per epoch
        :param log_dir: the directory that contains debug information
        :return: a ``BaseAgent``
        """
        return BaseAgent(network=network, environment=environment, num_of_threads=num_of_threads,
                         save_frequency=checkpoint_frequency, report_frequency=learner_report_frequency,
                         num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)

    @staticmethod
    def create(agent_type, policy_network, environment, num_of_learners=None, checkpoint_frequency=5e4,
               learner_report_frequency=10, num_of_epochs=50, steps_per_epoch=1e6,
               log_dir="./train/a3c", **args):
        """
        Instantiate a set of user-defined learners.

        :param agent_type: a learner defined by users
        :param policy_network: a reference to the ``PolicyNetwork``
        :param environment: is a subclass of BaseEnvironment
        :param num_of_learners: the number of learners used in the algorithm
        :param checkpoint_frequency: checkpoints will be saved with ``checkpoint_frequency``
        :param learner_report_frequency: each learner generates a debug message with ``learner_report_frequency``
        :param num_of_epochs: the number of training epochs
        :param steps_per_epoch: the number of training steps per epoch
        :param log_dir: the directory that contains checkpoints
        :param args: other args for the specified learner
        :return: the current agent
        """
        if num_of_learners is None:
            num_of_learners = agent_type.get_default_number_of_learners()
        agent = AgentFactory().get_base_agent(network=policy_network, environment=environment, num_of_threads=num_of_learners,
                                              checkpoint_frequency=checkpoint_frequency,
                                              learner_report_frequency=learner_report_frequency,
                                              num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch,
                                              log_dir=log_dir)

        thread_pool = [agent_type(agent, 'Learner: ' + str(t), agent.env_pool[t], policy_network,
                                  agent.thread_host.shared_dict, learner_report_frequency,
                                  **args)
                       for t in range(agent.num_of_threads)]

        agent.set_learners(thread_pool)

        return agent