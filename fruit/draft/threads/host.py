import numpy as np
import tensorflow as tf
import time
import collections as cl
import os
import psutil
import pickle


class Monitor:
    def __init__(self, agent, network, log_dir, save_interval=1e4, max_training_epochs=100, steps_per_epoch=1e6,
                 multi_objectives=False, multi_agents=False, recent_rewards=100, idle_time=1):
        self.log_dir = log_dir
        self.network = network
        self.num_epochs = max_training_epochs
        self.epoch_steps = steps_per_epoch
        self.save_interval = save_interval
        self.total_steps = self.num_epochs * self.epoch_steps
        self.save_interval = save_interval/self.epoch_steps
        self.agent = agent
        self.save_time_based = agent.save_time_based
        self.recent_rewards = cl.deque(maxlen=recent_rewards)
        self.multi_objs = multi_objectives
        self.idle_time = idle_time
        self.summary_writer = None
        self.multi_agents = multi_agents

        if self.network is not None:
            self.summary_writer = tf.summary.FileWriter(log_dir, graph=self.network.tf_graph)

            if not self.multi_objs:
                with self.network.get_graph().as_default():
                    self.reward = tf.placeholder(tf.int32)
                    self.reward_summary = tf.summary.scalar('reward', self.reward)

                with self.network.get_graph().as_default():
                    self.eval_reward = tf.placeholder(tf.int32)
                    self.eval_reward_summary = tf.summary.scalar('eval-reward', self.eval_reward)
        else:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.shared_dict = {
            "counter": 0,
            "done": False,
            "reward_list": [],
            "write_summaries_this_step": False,
            "summary_writer": self.summary_writer,
            "add_reward": self.__add_reward
        }

    def __add_reward(self, r, episode_steps):
        if self.network is not None:
            if not self.multi_objs:
                summary = self.network.tf_session.run(self.reward_summary, feed_dict={self.reward: r})
                self.summary_writer.add_summary(summary, global_step=self.shared_dict['counter'])
        self.shared_dict['reward_list'].append([r, self.shared_dict['counter'], episode_steps])
        self.recent_rewards.append(r)

    def __print_log(self, time_diff, current_epoch):
        print('================================================================'
              '==========================================================')
        print('Time elapse: {0:.2f} minutes'.format(time_diff / 60))
        print('Total steps/Total epoch: {0}/{1:.2f}'.format(self.shared_dict['counter'], current_epoch))
        print('Steps Per Second: {0:.4f}'.format(self.shared_dict['counter'] / time_diff))

        if not self.multi_objs:
            if len(self.shared_dict['reward_list']) > 0:
                print('Max Reward:', np.max([x[0] for x in self.shared_dict['reward_list']]))
                print('Mean reward:', np.mean([x[0] for x in self.shared_dict['reward_list']]))
                print('Mean-100 Reward: {0:.4f}'.format(np.mean([x for x in self.recent_rewards])))
                print('Max-100/Min-100 Reward: {0}/{1}'.format(np.max([x for x in self.recent_rewards]),
                                                           np.min([x for x in self.recent_rewards])))
        else:
            if len(self.shared_dict['reward_list']) > 0:
                # print([x[0] for x in self.shared_dict['reward_list']])
                print('Max Reward:', np.max([x[0] for x in self.shared_dict['reward_list']], axis=0))
                print('Mean reward:', np.mean([x[0] for x in self.shared_dict['reward_list']], axis=0))

        if not self.agent.is_testing_mode:
            print('Learning Rate:', self.agent.current_learning_rate)

        print('=================================================================='
              '========================================================')

        if self.network is not None:
            if not self.agent.is_testing_mode:
                self.shared_dict['write_summaries_this_step'] = True

            self.network.save_model(self.log_dir + 'model', global_step=self.shared_dict['counter'])
        else:
            _str = self.log_dir + 'checkpoint_' + str(self.shared_dict['counter'])
            print("Checkpoint saved: " + _str)
            self.agent.save_model(_str)

    def run_epochs_mac(self, learners):
        st = time.time()
        last_save = 0
        num_of_checkpoints = 0
        try:
            while (self.agent.checkpoint_stop == 0 and self.shared_dict['counter'] < self.total_steps) or \
                    (self.agent.checkpoint_stop > 0 and num_of_checkpoints < self.agent.checkpoint_stop):

                if self.agent.converged:
                    break
                if self.save_time_based <= 0:
                    current_epoch = self.shared_dict['counter'] / self.epoch_steps
                    if current_epoch > last_save + self.save_interval:
                        et = time.time()

                        num_of_checkpoints = num_of_checkpoints + 1
                        self.__print_log(et-st, current_epoch)

                        last_save = current_epoch
                else:
                    current_time = time.time()
                    diff = (current_time - st)/60
                    if diff > last_save + self.save_time_based:
                        et = time.time()
                        current_epoch = self.shared_dict['counter'] / self.epoch_steps

                        num_of_checkpoints = num_of_checkpoints + 1
                        self.__print_log(et-st, current_epoch)

                        last_save = diff

                learners[0].run()
        except KeyboardInterrupt:
            print('User stops, stopping all threads...')

        self.shared_dict['done'] = True
        return self.shared_dict['reward_list']

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
        num_of_checkpoints = 0
        try:
            while (self.agent.checkpoint_stop == 0 and self.shared_dict['counter'] < self.total_steps) or \
                    (self.agent.checkpoint_stop > 0 and num_of_checkpoints < self.agent.checkpoint_stop):

                if self.agent.converged:
                    break
                if self.save_time_based <= 0:
                    current_epoch = self.shared_dict['counter'] / self.epoch_steps
                    if current_epoch > last_save + self.save_interval:
                        et = time.time()

                        num_of_checkpoints = num_of_checkpoints + 1
                        self.__print_log(et-st, current_epoch)

                        last_save = current_epoch

                        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
                        current_mem = psutil.virtual_memory().used
                        ram_stats.append(current_mem)
                        cpu_stats.append(cpu_usage)
                        print("CPU STATS:", cpu_usage, current_mem)
                else:
                    current_time = time.time()
                    diff = (current_time - st)/60
                    if diff > last_save + self.save_time_based:
                        et = time.time()
                        current_epoch = self.shared_dict['counter'] / self.epoch_steps

                        num_of_checkpoints = num_of_checkpoints + 1
                        self.__print_log(et-st, current_epoch)

                        last_save = diff

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

        self.shared_dict['done'] = True
        for t in threads:
            t.join()
        print('All threads stopped')
        return self.shared_dict['reward_list']
