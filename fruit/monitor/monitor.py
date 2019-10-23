import threading

import numpy as np
import tensorflow as tf
import time
import collections as cl
import os
import psutil
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm


class AgentMonitor:
    Q_ADD_REWARD = 'add_reward'
    Q_REWARD_LIST = 'reward_list'
    Q_WRITER = 'writer'
    Q_FINISH = 'done'
    Q_GLOBAL_STEPS = 'counter'
    Q_LOGGING = 'logging'
    Q_LEARNING_RATE = 'learning_rate'
    Q_LOCK = 'thread_lock'

    def __init__(self, agent, network, log_dir, save_interval=1e4, max_training_epochs=100, steps_per_epoch=1e6,
                 number_of_objectives=1, recent_rewards=100, idle_time=1):
        self.log_dir = log_dir
        self.network = network
        self.num_epochs = max_training_epochs
        self.epoch_steps = steps_per_epoch
        self.save_interval = save_interval
        self.total_steps = self.num_epochs * self.epoch_steps
        self.save_interval = save_interval/self.epoch_steps
        self.agent = agent
        self.recent_rewards = cl.deque(maxlen=recent_rewards)
        self.number_of_objectives = number_of_objectives
        self.idle_time = idle_time
        self.summary_writer = None
        self.thread_lock = threading.Lock()

        if self.network is not None:
            self.summary_writer = tf.summary.FileWriter(log_dir, graph=self.network.tf_graph)

            if self.number_of_objectives <= 1:
                with self.network.get_graph().as_default():
                    self.reward = tf.placeholder(tf.int32)
                    self.reward_summary = tf.summary.scalar('reward', self.reward)
            else:
                with self.network.get_graph().as_default():
                    self.reward = [tf.placeholder(tf.int32) for _ in range(self.number_of_objectives)]
                    self.reward_summary = [tf.summary.scalar('reward_' + str(i), self.reward[i])
                                           for i in range(self.number_of_objectives)]
        else:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.shared_dict = {
            AgentMonitor.Q_GLOBAL_STEPS: 0,
            AgentMonitor.Q_FINISH: False,
            AgentMonitor.Q_REWARD_LIST: [],
            AgentMonitor.Q_LOGGING: False,
            AgentMonitor.Q_WRITER: self.summary_writer,
            AgentMonitor.Q_ADD_REWARD: self.__add_reward,
            AgentMonitor.Q_LEARNING_RATE: self.network.get_config().
                get_initial_learning_rate() if self.network is not None else None,
            AgentMonitor.Q_LOCK: self.thread_lock
        }

    def __add_reward(self, r, episode_steps):
        if self.network is not None:
            if self.number_of_objectives <= 1:
                summary = self.network.tf_session.run(self.reward_summary, feed_dict={self.reward: r})
                self.summary_writer.add_summary(summary, global_step=self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS])
            else:
                for i in range(self.number_of_objectives):
                    summary = self.network.tf_session.run(self.reward_summary[i], feed_dict={self.reward[i]: r[i]})
                    self.summary_writer.add_summary(summary, global_step=self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS])
        self.shared_dict[AgentMonitor.Q_REWARD_LIST].append([r, self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS],
                                                             episode_steps])
        self.recent_rewards.append(r)

    def __print_log(self, time_diff, current_epoch):
        print('================================================================'
              '==========================================================')
        print('Time elapse: {0:.2f} minutes'.format(time_diff / 60))
        print('Total steps/Total epoch: {0}/{1:.2f}'.format(self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS],
                                                            current_epoch))
        print('Steps Per Second: {0:.4f}'.format(self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS] / time_diff))
        plt.figure(figsize=(6, 4))
        if self.number_of_objectives <= 1:
            if len(self.shared_dict[AgentMonitor.Q_REWARD_LIST]) > 0:
                print('Max Reward:', np.max([x[0] for x in self.shared_dict[AgentMonitor.Q_REWARD_LIST]]))
                print('Mean reward:', np.mean([x[0] for x in self.shared_dict[AgentMonitor.Q_REWARD_LIST]]))
                print('Mean-100 Reward: {0:.4f}'.format(np.mean(self.recent_rewards)))
                print('Max-100/Min-100 Reward: {0}/{1}'.format(np.max(self.recent_rewards),
                                                               np.min(self.recent_rewards)))
                reward_list = self.shared_dict[AgentMonitor.Q_REWARD_LIST]
                plt.scatter([x[1] for x in reward_list], [x[0] for x in reward_list], marker='.',
                            alpha=0.6, c=range(len([x[1] for x in reward_list])), s=1,
                            cmap=cm.get_cmap('viridis_r', len([x[1] for x in reward_list])))
                plt.savefig(self.log_dir + 'reward_distribution_' + str(self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS])
                            + '.pdf')
        else:
            if len(self.shared_dict[AgentMonitor.Q_REWARD_LIST]) > 0:
                print('Max Reward:', np.max([x[0][0] for x in self.shared_dict[AgentMonitor.Q_REWARD_LIST]], axis=0))
                print('Mean reward:', np.mean([x[0][0] for x in self.shared_dict[AgentMonitor.Q_REWARD_LIST]], axis=0))
                print('Mean-100 Reward: {0:.4f}'.format(np.mean([x[0] for x in self.recent_rewards])))
                print('Max-100/Min-100 Reward: {0}/{1}'.format(np.max([x[0] for x in self.recent_rewards]),
                                                               np.min([x[0] for x in self.recent_rewards])))
                reward_list = self.shared_dict[AgentMonitor.Q_REWARD_LIST]
                plt.scatter([x[1] for x in reward_list], [x[0][0] for x in reward_list], marker='.',
                            alpha=0.6, c=range(len([x[1] for x in reward_list])), s=1,
                            cmap=cm.get_cmap('viridis_r', len([x[1] for x in reward_list])))
                plt.savefig(self.log_dir + 'reward_distribution_' + str(self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS])
                            + '.pdf')
        plt.close()

        with open(self.log_dir + 'reward_distribution_' +
                  str(self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS]) + '.p', 'wb+') as file:
            pickle.dump([self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS], self.shared_dict[AgentMonitor.Q_REWARD_LIST]],
                        file)

        if not self.agent.is_testing_mode:
            print('Learning Rate:', self.shared_dict[AgentMonitor.Q_LEARNING_RATE])

        print('=================================================================='
              '========================================================')

        if not self.agent.is_testing_mode:
            self.shared_dict[AgentMonitor.Q_LOGGING] = True

        if self.network is not None:
            self.network.save_model(self.log_dir + 'model', global_step=self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS])
        else:
            _str = self.log_dir + 'checkpoint_' + str(self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS])
            self.agent.thread_pool[0].save_model(_str)

    def run_epochs_mac(self, learners):
        st = time.time()
        last_save = 0
        try:
            while self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS] < self.total_steps:
                if self.agent.converged:
                    break

                current_epoch = self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS] / self.epoch_steps
                if current_epoch > last_save + self.save_interval:
                    et = time.time()

                    self.__print_log(et-st, current_epoch)

                    last_save = current_epoch

                learners[0].run()
        except KeyboardInterrupt:
            print('User stops, stopping all threads...')

        self.shared_dict[AgentMonitor.Q_FINISH] = True

        return self.shared_dict[AgentMonitor.Q_REWARD_LIST]

    def run_epochs(self, learners):
        threads = []
        st = time.time()

        cpu_stats = []
        ram_stats = []
        current_mem = psutil.virtual_memory().used
        ram_stats.append(current_mem)

        for thread in learners:
            thread.start()
            threads.append(thread)

        last_save = 0

        try:
            while self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS] < self.total_steps:
                if self.agent.converged:
                    break

                current_epoch = self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS] / self.epoch_steps
                if current_epoch > self.save_interval + last_save:
                    et = time.time()

                    self.__print_log(et-st, current_epoch)

                    last_save = current_epoch

                    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
                    current_mem = psutil.virtual_memory().used
                    ram_stats.append(current_mem)
                    cpu_stats.append(cpu_usage)
                    print("CPU STATS:", cpu_usage, current_mem)

                if self.idle_time > 0:
                    time.sleep(self.idle_time)
        except KeyboardInterrupt:
            print('User stops, stopping all threads...')

        file = open(self.log_dir + '/cpu_stats.p', 'wb+')
        pickle.dump(cpu_stats, file)
        file.close()

        file = open(self.log_dir + '/ram_stats.p', 'wb+')
        pickle.dump(ram_stats, file)
        file.close()

        self.shared_dict[AgentMonitor.Q_FINISH] = True
        for t in threads:
            t.join()

        current_epoch = self.shared_dict[AgentMonitor.Q_GLOBAL_STEPS] / self.epoch_steps
        et = time.time()
        self.__print_log(et - st, current_epoch)

        print('All threads stopped')
        return self.shared_dict[AgentMonitor.Q_REWARD_LIST]
