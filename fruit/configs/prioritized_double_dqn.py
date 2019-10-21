from fruit.configs.double_dqn import AtariDoubleDQNConfig
import tensorflow as tf
import numpy as np


class PrioritizedAtariDQNConfig(AtariDoubleDQNConfig):
    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_targets, \
            self.tf_inputs_norm, self.tf_is_weights = self.create_input()

        with tf.variable_scope('network'):
            self.tf_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('target'):
            self.tf_target_output = self.create_network()
            self.tf_target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        with tf.name_scope('loss'):
            self.tf_est_q_values, self.tf_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        targets = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_targets")
        weights = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_is_weights")
        return inputs, actions, targets, inputs_norm, weights

    def create_loss(self):
        with tf.name_scope('estimate_q_values'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            estimate_q_values = tf.reduce_sum(tf.multiply(self.tf_output, actions_one_hot), axis=1)

        with tf.name_scope('loss'):
            loss = tf.squared_difference(estimate_q_values, self.tf_targets) * self.tf_is_weights

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_mean(loss)
            tf.summary.scalar('total-loss', total_loss)

        return estimate_q_values, loss, total_loss

    def get_params(self, data_dict):
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminals = data_dict['terminals']
        learning_rate = data_dict['learning_rate']
        logging = data_dict['logging']
        global_step = data_dict['global_step']
        is_weights = data_dict['is_weights']
        return states, actions, rewards, next_states, terminals, learning_rate, logging, global_step, is_weights

    def train(self, session, data_dict):
        states, actions, rewards, next_states, terminals, learning_rate, logging, global_step, is_weights = \
            self.get_params(data_dict)

        if global_step % self.target_update == 0:
            self.update_target_network(session)

        batch_size = np.array(states).shape[0]

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = session.run(self.tf_output, feed_dict=feed_dict)
        max_estimate_actions = np.argmax(q_values_next, axis=1)

        q_values_target_next = session.run(self.tf_target_output, feed_dict=feed_dict)

        target_q_values_next = q_values_target_next[np.arange(batch_size), max_estimate_actions]

        targets = rewards + np.subtract(1., terminals) * self.gamma * target_q_values_next

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions,
                     self.tf_targets: targets, self.tf_is_weights: is_weights}

        if logging:
            if self.debug_mode:
                print("##############################################################################")
                print("IS WEIGHTS:")
                print(is_weights)
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", self.gamma)
                q_values, loss, total_loss = \
                    session.run([self.tf_est_q_values, self.tf_loss, self.tf_total_loss], feed_dict=feed_dict)
                print("Q VALUES:")
                print(q_values)
                print("LOSS:")
                print(loss)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_td_errors(self, session, data_dict):
        states, actions, rewards, next_states, terminals, _, _, _, _ = \
            self.get_params(data_dict)

        batch_size = np.array(states).shape[0]

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = session.run(self.tf_output, feed_dict=feed_dict)
        max_estimate_actions = np.argmax(q_values_next, axis=1)

        q_values_target_next = session.run(self.tf_target_output, feed_dict=feed_dict)

        target_q_values_next = q_values_target_next[np.arange(batch_size), max_estimate_actions]

        targets = rewards + np.subtract(1., terminals) * self.gamma * target_q_values_next

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_targets: targets}
        estimate_q_values = session.run(self.tf_est_q_values, feed_dict=feed_dict)

        return np.abs(targets - estimate_q_values)
