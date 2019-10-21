class MOA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch,
                                   multi_objectives=True
                                   )

        self.thread_pool = [MOA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]


class JairA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch,
                                   multi_objectives=True
                                   )

        self.thread_pool = [JairA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon,
                                   thread_id=t, total_weights=len(network.get_config().weights))
                            for t in range(self.num_of_threads)]


class MAA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        print("Number of threads: ", self.num_of_threads)

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch
                                   )

        self.thread_pool = [MAA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                        global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                        reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                        report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]


class MapMAA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        print("Number of threads: ", self.num_of_threads)

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch
                                   )

        self.thread_pool = [MapMAA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                           global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                           reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                           report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]


class RiverraidA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch
                                   )

        self.thread_pool = [RiverraidA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]


class NIPSA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch
                                   )

        self.thread_pool = [NIPSA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]


class MANIPSA3CAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0,
                 ):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch
                                   )

        self.thread_pool = [MANIPSA3CWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]


class A3CLSTMAgent(BaseAgent):
    def __init__(self, network, environment, num_of_threads=0, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/a3c", update_network_frequency=5, reward_clips=[-1,1], using_e_greedy=False,
                 anneal_learning_rate=True, initial_epsilon=0.05, report_frequency=10, save_time_based=0, checkpoint_stop=0):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        if self.num_of_threads <= 0:
            self.num_of_threads = multiprocessing.cpu_count()

        self.t_max = update_network_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.thread_host = Monitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                   max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch)

        self.thread_pool = [A3CLSTMWorker(self, name='Worker: ' + str(t), environment=self.env_pool[t], network=network,
                                   global_dict=self.thread_host.shared_dict, async_update_steps=self.t_max,
                                   reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                   report_frequency=report_frequency, epsilon_annealing_start=initial_epsilon)
                            for t in range(self.num_of_threads)]