from fruit.agents.base import BaseAgent
import datetime
from fruit.threads.learner import LSTMWorker
from fruit.threads.host import AgentMonitor


class A3CLSTMAgent(BaseAgent):
    def __init__(self, num_of_threads, network, environment, save_frequency=5e4,
                 log_dir="./train/na3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False):

        super().__init__(network, environment)

        self.num_of_threads = num_of_threads
        self.num_of_epochs = network.max_training_epochs
        self.save_interval = save_frequency
        #run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.log_dir = log_dir# + '_' + run_date + '/'
        self.t_max = update_network_frequency
        self.reward_clips = reward_clips
        self.using_e_greedy = using_e_greedy

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = AgentMonitor(network, log_dir=self.log_dir, save_interval=save_frequency)

        self.thread_pool = [LSTMWorker(name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=10)
                            for t in range(self.num_of_threads)]

    def train(self):

        reward_list = self.thread_host.run_epochs(self.thread_pool)

        import matplotlib.pyplot as plt
        plt.plot([x[1] for x in reward_list], [x[0] for x in reward_list], '.')
        plt.savefig(self.log_dir + 'rewards.png')
        plt.show()

        return max([x[0] for x in reward_list])

    def evaluate(self):
        return None