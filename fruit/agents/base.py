import datetime
from fruit.monitor.monitor import AgentMonitor
import platform


class BaseAgent(object):
    def __init__(self, network, environment, num_of_threads=1, num_of_epochs=100, steps_per_epoch=1e6,
                 log_dir="./log/", report_frequency=1, save_frequency=5e4):
        self.network = network
        self.env = environment
        self.num_of_threads = num_of_threads
        self.num_of_epochs = num_of_epochs
        self.steps_per_epoch = steps_per_epoch
        run_date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.log_dir = log_dir + '_' + run_date + '/'
        self.report_frequency = report_frequency
        self.save_interval = save_frequency
        self.thread_pool = None

        self.max_training_steps = num_of_epochs * steps_per_epoch
        self.is_testing_mode = True
        self.converged = False

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = AgentMonitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                        max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch,
                                        number_of_objectives=environment.get_number_of_objectives()
                                        )

    def set_learners(self, learners):
        self.thread_pool = learners

    def get_log_dir(self):
        return self.log_dir

    def train(self):
        if self.thread_pool is None or self.thread_host is None:
            raise ValueError("No definition of worker!!")

        if self.network is not None:
            self.network.set_save_model(True)

        self.is_testing_mode = False

        reward_list = self.thread_host.run_epochs(self.thread_pool)

        return reward_list

    def evaluate(self):
        if self.thread_pool is None or self.thread_host is None:
            raise ValueError("No definition of worker!!")

        if self.network is not None:
            self.network.set_save_model(False)

        self.is_testing_mode = True

        if self.env.is_render and self.num_of_threads == 1 and platform.system() == 'Darwin':
            reward_list = self.thread_host.run_epochs_mac(self.thread_pool)
        else:
            reward_list = self.thread_host.run_epochs(self.thread_pool)

        return reward_list


