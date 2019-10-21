from fruit.agents.base import BaseAgent
from fruit.threads.learner import ExpReplayWorker, MOExpReplayWorker
from fruit.threads.host import AgentMonitor
from fruit.buffers.replay import AsyncExperienceReplay
from fruit.buffers.replay import SyncExperienceReplay
from fruit.buffers.tree import SyncSumTree
from fruit.threads.learner import PrioritizedExpReplayWorker
import psutil


class DQNAgent(BaseAgent):
    def __init__(self, network, environment, save_frequency=5e5, num_of_epochs=10, steps_per_epoch=1e6,
                 log_dir="./train/dqn", train_frequency=4, batch_size=32, warmup_steps = 50000,
                 reward_clips=[-1,1], using_e_greedy=True, anneal_learning_rate=False,
                 initial_epsilon=1, final_epsilon=0.1, epsilon_anneal_steps=1000000, report_frequency=10,
                 save_time_based=0, checkpoint_stop=0, num_of_threads=1, exp_replay_size=900000, prioritized=False,
                 prioritized_alpha=0.6, importance_sampling=False):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop,
                         importance_sampling=importance_sampling)

        self.train_frequency = train_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        recommended_exp_size = int(1000000 * total_memory/32)
        if exp_replay_size > recommended_exp_size:
            exp_replay_size = recommended_exp_size
            print("Recommended experience replay size based on memory constraints: " + str(exp_replay_size))

        if num_of_threads <= 1:
            if prioritized:
                self.replay = SyncSumTree(alpha=prioritized_alpha, size=exp_replay_size,
                                          state_history=network.get_config().get_history_length(), debug=False)
            else:
                self.replay = SyncExperienceReplay(exp_replay_size, network.get_config().get_history_length())
        else:
            if prioritized:
                raise ValueError('Prioritized experience replay with async is not supported !')
            else:
                self.replay = AsyncExperienceReplay(exp_replay_size)

        self.thread_host = AgentMonitor(self, network=network, log_dir=self.log_dir, save_interval=save_frequency,
                                        max_training_epochs=self.num_of_epochs, steps_per_epoch=self.steps_per_epoch)

        if prioritized:
            self.thread_pool = [PrioritizedExpReplayWorker(self, name='PriExpReplayWorker: ' + str(t),
                                                           environment=self.env_pool[t], network=network,
                                                           global_dict=self.thread_host.shared_dict,
                                                           replay=self.replay,
                                                           async_update_steps=self.train_frequency,
                                                           batch_size=batch_size, warmup_steps=warmup_steps,
                                                           reward_clip_vals=self.reward_clips,
                                                           using_e_greedy=self.using_e_greedy,
                                                           report_frequency=report_frequency,
                                                           epsilon_annealing_start=initial_epsilon,
                                                           epsilon_annealing_choices=[final_epsilon],
                                                           epsilon_annealing_probabilities=[1.0],
                                                           epsilon_annealing_steps=epsilon_anneal_steps
                                                           )
                                for t in range(self.num_of_threads)]
        else:
            self.thread_pool = [ExpReplayWorker(self, name='ExpReplayWorker: ' + str(t), environment=self.env_pool[t],
                                                network=network, global_dict=self.thread_host.shared_dict,
                                                replay=self.replay,
                                                async_update_steps=self.train_frequency,
                                                batch_size=batch_size, warmup_steps=warmup_steps,
                                                reward_clip_vals=self.reward_clips, using_e_greedy=self.using_e_greedy,
                                                report_frequency=report_frequency,
                                                epsilon_annealing_start=initial_epsilon,
                                                epsilon_annealing_choices=[final_epsilon],
                                                epsilon_annealing_probabilities=[1.0],
                                                epsilon_annealing_steps=epsilon_anneal_steps)
                            for t in range(self.num_of_threads)]


class MODQNAgent(BaseAgent):
    def __init__(self, network, environment, save_frequency=5e4, num_of_epochs=50, steps_per_epoch=1e6,
                 log_dir="./train/dqn", train_frequency=4, batch_size=32, warmup_steps = 50000,
                 reward_clips=[-1,1], using_e_greedy=True, anneal_learning_rate=False,
                 initial_epsilon=1, final_epsilon=0.1, epsilon_anneal_steps=1000000, report_frequency=10,
                 save_time_based=0, checkpoint_stop=0, num_of_threads=1, exp_replay_size=900000,
                 weights=None):

        super().__init__(network, environment, num_of_threads=num_of_threads, num_of_epochs=num_of_epochs,
                         steps_per_epoch=steps_per_epoch, log_dir=log_dir, reward_clips=reward_clips,
                         using_e_greedy=using_e_greedy, report_frequency=report_frequency,
                         save_frequency=save_frequency, anneal_learning_rate=anneal_learning_rate,
                         save_time_based=save_time_based, checkpoint_stop=checkpoint_stop)

        self.train_frequency = train_frequency

        self.env_pool = [self.env.clone() for _ in range(self.num_of_threads)]

        self.weights = weights

        num_of_exps = len(self.weights)
        size_of_each_exps = int(exp_replay_size/num_of_exps)

        self.replay = []

        for i in range(num_of_exps):
            self.replay.append(AsyncExperienceReplay(size_of_each_exps))

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
                                            epsilon_annealing_steps=epsilon_anneal_steps,
                                            weights=self.weights,
                                            id=t,
                                            num_of_threads=num_of_threads
                                            )
                            for t in range(self.num_of_threads)]