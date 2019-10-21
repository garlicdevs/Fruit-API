from fruit.utils.annealer import Annealer
from fruit.threads.molearner import MOQWorker
from fruit.buffers.table import TLOLookupTable, LinearLookupTable
from fruit.agents.base import BaseAgent
from fruit.threads.host import AgentMonitor
from fruit.buffers.replay import SyncExperienceReplay
from fruit.threads.molearner import MOExpReplayWorker


class MOQAgent(BaseAgent):
    def __init__(self, environment, num_of_epochs=10, steps_per_epoch=100000,
                 log_dir='./train/moq', using_e_greedy=True, report_frequency=100,
                 summary_frequency=900000, discounted_factor=0.9, learning_rate=0.9, traces_factor=0.9, batch_size=5,
                 epsilon_annealing_start=0.9, load_model_path=None, thresholds=None, target_reward=None, is_linear=False):
        super().__init__(None, environment, num_of_threads=1, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, using_e_greedy=using_e_greedy,
                         anneal_learning_rate=False, report_frequency=report_frequency,
                         save_frequency=summary_frequency
                         )

        self.initial_learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.load_model_path = load_model_path

        self.gamma = discounted_factor
        self.traces_factor = traces_factor

        self.is_linear = is_linear

        # Disable annealing learning rate
        self.learning_rate_annealer = Annealer(self.initial_learning_rate, 0, None)

        self.num_of_objectives = environment.get_num_of_objectives()

        self.init_q_values = [0.] * self.num_of_objectives
        self.thresholds = [0.] * (self.num_of_objectives - 1)

        if not is_linear:
            self.table = TLOLookupTable(environment=environment, init_value=0., thresholds=self.thresholds)
        else:
            self.table = LinearLookupTable(environment=environment, init_value=0., thresholds=self.thresholds)

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = AgentMonitor(self, network=None, log_dir=self.log_dir, save_interval=summary_frequency,
                                        max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch,
                                        multi_objectives=True, idle_time=0)

        self.thread_pool = [MOQWorker(self, name='MOQWorker: ' + str(t), environment=self.env_pool[t],
                                      global_dict=self.thread_host.shared_dict, num_of_objs=self.num_of_objectives,
                                      async_update_steps=1,
                                      using_e_greedy=self.using_e_greedy,
                                      report_frequency=report_frequency,
                                      epsilon_annealing_start=epsilon_annealing_start,
                                      epsilon_annealing_choices=[0],
                                      epsilon_annealing_probabilities=[1.0],
                                      epsilon_annealing_steps=num_of_epochs * steps_per_epoch,
                                      global_epsilon_annealing=True,
                                      gamma=discounted_factor,
                                      traces_factor=traces_factor,
                                      batch_size=batch_size,
                                      load_model_path=load_model_path,
                                      lookup_table=self.table,
                                      thresholds=thresholds,
                                      target_reward=target_reward,
                                      is_linear=is_linear
                                      )
                                for t in range(self.num_of_threads)]

    def train(self):
        if self.thread_pool is None or self.thread_host is None:
            raise ValueError("No definition of worker!!")

        self.is_testing_mode = False

        reward_list = self.thread_host.run_epochs(self.thread_pool)

        return reward_list

    def evaluate(self):
        if self.thread_pool is None or self.thread_host is None:
            raise ValueError("No definition of worker!!")

        self.is_testing_mode = True

        reward_list = self.thread_host.run_epochs(self.thread_pool)

        return reward_list

    def load_model(self):
        self.table.load_value_function(self.load_model_path)
        print("Load values:")
        self.table.print_values()

    def save_model(self, file_name):
        print("Save values:")
        self.table.print_values()
        self.table.save_value_function(file_name)


class MODQNAgent(BaseAgent):
    def __init__(self, network, environment, save_frequency=50000, num_of_epochs=10, steps_per_epoch=100000,
                 log_dir="./train/dqn", train_frequency=1, batch_size=32, warmup_steps = 50000,
                 using_e_greedy=True, anneal_learning_rate=False,
                 initial_epsilon=1, final_epsilon=0.1, epsilon_anneal_steps=500000, report_frequency=100,
                 save_time_based=0, checkpoint_stop=0, num_of_threads=1, exp_replay_size=500000):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=None,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        self.train_frequency = train_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.replay = SyncExperienceReplay(exp_replay_size)

        self.thread_host = AgentMonitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                        max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch,
                                        multi_objectives=True)

        self.thread_pool = [MOExpReplayWorker(self, name='ExpReplayWorker: ' + str(t), environment=self.env_pool[t],
                                            network=network,
                                            global_dict=self.thread_host.shared_dict,
                                            replay=self.replay,
                                            async_update_steps=self.train_frequency,
                                            batch_size=batch_size, warmup_steps=warmup_steps,
                                            reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                            report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon,
                                            epsilon_annealing_choices=[final_epsilon],
                                            epsilon_annealing_probabilities=[1.0],
                                            epsilon_annealing_steps=epsilon_anneal_steps)
                            for t in range(self.num_of_threads)]