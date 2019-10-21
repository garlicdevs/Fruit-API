from fruit.networks.base import BaseNetwork
import tensorflow as tf
import numpy as np
from copy import deepcopy


class A3CLSTMNetwork(BaseNetwork):
    def __init__(self, network_config, optimizer=tf.train.RMSPropOptimizer(learning_rate=0.004, decay=0.99, epsilon=0.1),
                 gamma=0.99, beta=0.0001, global_norm_clipping=40, initial_learning_rate=0.004,
                 max_training_epochs = 40, steps_per_epoch=1e6, using_gpu=True,
                 anneal_learning_rate=True, stochastic_policy=True):

        if network_config is None:
            raise ValueError("A3CNetwork needs a network_config to work !")

        self.network_config = network_config
        self.beta = beta
        self.global_norm_clipping = global_norm_clipping
        self.gamma = gamma
        self.network_config = network_config

        self.prev_lstm_state = None
        self.prev_lstm_state_1 = None

        super().__init__(optimizer=optimizer,
                         initial_learning_rate=initial_learning_rate,
                         max_training_epochs=max_training_epochs, steps_per_epoch=steps_per_epoch,
                         using_gpu=using_gpu,
                         anneal_learning_rate=anneal_learning_rate, stochastic_policy=stochastic_policy)

        self.reset_lstm_state()

    def create_network(self):

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards = self.network_config.create_input()

        with tf.variable_scope('network'):

            self.tf_actor_output, self.tf_critic_output, self.initial_lstm_state, \
            self.new_lstm_state, self.initial_lstm_state_1, self.new_lstm_state_1, \
                = self.network_config.create_network(self.tf_inputs)

            tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            with tf.name_scope('critic-reward-diff'):
                self.critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

            with tf.name_scope('log-of-actor-policy'):

                self.actions_one_hot = tf.one_hot(self.tf_actions, depth=self.network_config.get_output_size(), name='one-hot',
                                               on_value=1.0, off_value=0.0, dtype=tf.float32)
                self.log_policy = tf.log(self.tf_actor_output + 1e-6)
                self.log_policy_one_hot = tf.multiply(self.log_policy, self.actions_one_hot)
                self.log_policy_action = tf.reduce_sum(self.log_policy_one_hot, axis=1)

            with tf.name_scope('actor-entropy'):
                self.actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, self.log_policy))
                tf.summary.scalar("actor-entropy", self.actor_entropy)

            with tf.name_scope('actor-loss'):
                self.actor_loss = tf.reduce_sum(tf.multiply(self.log_policy_action, tf.stop_gradient(self.critic_diff)))
                tf.summary.scalar('actor-loss', self.actor_loss)

            with tf.name_scope('critic-loss'):
                self.critic_loss = tf.nn.l2_loss(self.critic_diff) * 0.5
                tf.summary.scalar('critic-loss', self.critic_loss)

            with tf.name_scope('total-loss'):
                self.total_loss = tf.reduce_sum(self.critic_loss + self.actor_loss + (self.actor_entropy * self.beta))
                tf.summary.scalar('total-loss', self.total_loss)

        with tf.name_scope('shared-optimizer'):
            self.tf_learning_rate = tf.placeholder(tf.float32)
            with tf.name_scope('compute-clip-grads'):
                gradients = self.optimizer.compute_gradients(self.total_loss)
                tensors = [tensor for gradient, tensor in gradients]
                grads = [gradient for gradient, tensor in gradients]
                clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
                clipped_grads_tensors = zip(clipped_gradients, tensors)
                self.tf_train_step = self.optimizer.apply_gradients(clipped_grads_tensors)

            tf.summary.scalar('learning-rate', self.tf_learning_rate)
            self.tf_summaries = tf.summary.merge_all()

        return tf_network_variables

    def get_output(self, state):
        if self.network_config.get_history_length() > 1:
            feed_dict = {self.tf_inputs: state, self.initial_lstm_state: self.prev_lstm_state}
        else:
            feed_dict = {self.tf_inputs: [state], self.initial_lstm_state: self.prev_lstm_state,
                         self.initial_lstm_state_1: self.prev_lstm_state_1}
        probs, new_state = self.tf_session.run([self.tf_actor_output, self.new_lstm_state], feed_dict=feed_dict)
        self.prev_lstm_state = new_state
        return self.get_action(probs[0])

    def debug(self, state):
        self.reset_lstm_state()
        if self.network_config.get_history_length() > 1:
            feed_dict = {self.tf_inputs: state, self.initial_lstm_state: self.prev_lstm_state}
        else:
            feed_dict = {self.tf_inputs: [state], self.initial_lstm_state: self.prev_lstm_state,
                         self.initial_lstm_state_1: self.prev_lstm_state_1}
        value = self.tf_session.run([self.tf_actor_output], feed_dict=feed_dict)
        return value

    def train_network(self, states, acts, rewards, states_target, terminals, global_step=0, summaries=False):
        self.anneal_learning_rate(global_step)

        if sum(terminals) > 1:
            raise ValueError('TD reward for mutiple terminal states in a batch is undefined')

        curr_reward = 0
        if not terminals[-1]:
            target_feed_dict = {self.tf_inputs: [states_target[-1]],
                                self.initial_lstm_state_1: self.prev_lstm_state_1}
            curr_reward, new_state = self.tf_session.run([self.tf_critic_output, self.new_lstm_state_1], feed_dict=target_feed_dict)
            self.prev_lstm_state_1 = new_state
            curr_reward = (max(curr_reward))
        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)


        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: acts, self.tf_rewards: td_rewards,
                     self.tf_learning_rate: self.current_learning_rate,
                     self.initial_lstm_state: self.prev_lstm_state, self.initial_lstm_state_1: self.prev_lstm_state_1}

        if summaries:
            print(list(states))
            print(acts)
            print(td_rewards)
            print(terminals)
            actor_output, critic_output, action_one_hot, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                self.tf_session.run([self.tf_actor_output, self.tf_critic_output,
                                     self.actions_one_hot, self.log_policy, self.log_policy_action,
                                     self.critic_diff, self.critic_loss, self.actor_loss, self.actor_entropy, self.total_loss,
                                     ], feed_dict=feed_dict)
            print(actor_output)
            print(critic_output)
            print(action_one_hot)
            print(log_policy)
            print(log_policy_action)
            print(critic_diff)
            print(critic_loss)
            print(actor_loss)
            print(actor_entropy)
            print(total_loss)
            return self.tf_session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return self.tf_session.run([self.tf_actor_output, self.tf_train_step], feed_dict=feed_dict)

    def reset_lstm_state(self, new_state=None, new_state_1=None):
        if new_state is not None:
            self.prev_lstm_state = new_state
            self.prev_lstm_state_1 = new_state_1
        else:
            self.prev_lstm_state = (np.zeros((1, 128)), np.zeros((1, 128)))
            self.prev_lstm_state_1 = (np.zeros((1, 128)), np.zeros((1, 128)))

    def get_actor_lstm_state(self):
        return deepcopy(self.prev_lstm_state)

    def get_critic_lstm_state(self):
        return deepcopy(self.prev_lstm_state_1)

