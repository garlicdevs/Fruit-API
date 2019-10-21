import multiprocessing
from fruit.agents.base import BaseAgent


class AgentFactory(object):
    @staticmethod
    def get_base_agent(network, environment, num_of_threads=0, checkpoint_frequency=5e4, learner_report_frequency=10,
                       num_of_epochs=50, steps_per_epoch=1e6, log_dir="./train/a3c"):
        return BaseAgent(network=network, environment=environment, num_of_threads=num_of_threads,
                         save_frequency=checkpoint_frequency, report_frequency=learner_report_frequency,
                         num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch, log_dir=log_dir)

    @staticmethod
    def create(agent_type, network, environment, num_of_learners=None, checkpoint_frequency=5e4,
               learner_report_frequency=10, num_of_epochs=50, steps_per_epoch=1e6,
               log_dir="./train/a3c", **args):
        if num_of_learners is None:
            num_of_learners = agent_type.get_default_number_of_learners()
        agent = AgentFactory().get_base_agent(network=network, environment=environment, num_of_threads=num_of_learners,
                                              checkpoint_frequency=checkpoint_frequency,
                                              learner_report_frequency=learner_report_frequency,
                                              num_of_epochs=num_of_epochs, steps_per_epoch=steps_per_epoch,
                                              log_dir=log_dir)

        thread_pool = [agent_type(agent, 'Learner: ' + str(t), agent.env_pool[t], network,
                                  agent.thread_host.shared_dict, learner_report_frequency,
                                  **args)
                       for t in range(agent.num_of_threads)]

        agent.set_learners(thread_pool)

        return agent