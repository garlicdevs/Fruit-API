from fruit.configs.base import Config
import tensorflow as tf
import numpy as np


class DQNConfig(Config):
    def __init__(self, environment, initial_learning_rate=0.00025, history_length=4, debug_mode=False,
                 discounted_factor=0.99, target_network_update=10000, optimizer='RMSProp-2'):
        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, debug_mode=debug_mode, gamma=discounted_factor,
                         optimizer=optimizer)
        self.target_update = target_network_update
        self.optimizer = self.optimizer(learning_rate=initial_learning_rate)


# Configuration for Atari games using DQN
class AtariDQNConfig(DQNConfig):
    def reset_config(self):
        return

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_targets, self.tf_inputs_norm = self.create_input()

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

    def get_target_network(self):
        return self.tf_inputs, self.tf_target_output, self.tf_target_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        targets = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_targets")
        return inputs, actions, targets, inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 32, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 64, 4, strides=2, activation_fn='relu', padding='valid',
                                                       scope='tf_layer_2')
        layer_3 = self.layer_manager.create_conv_layer(layer_2, 64, 3, strides=1, activation_fn='relu', padding='valid',
                                                       scope='tf_layer_3')
        layer_4 = self.layer_manager.create_fully_connected_layer(layer_3, 512, activation_fn='relu',
                                                                  scope='tf_layer_4')
        output = self.layer_manager.create_output(layer_4, self.output_size, scope='tf_output')
        return output

    def create_train_step(self):
        train_step = self.optimizer.minimize(self.tf_total_loss)
        summaries = tf.summary.merge_all()
        return summaries, train_step

    def create_loss(self):
        with tf.name_scope('estimate_q_values'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            estimate_q_values = tf.reduce_sum(tf.multiply(self.tf_output, actions_one_hot), axis=1)

        with tf.name_scope('loss'):
            loss = tf.squared_difference(estimate_q_values, self.tf_targets)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_mean(loss)
            tf.summary.scalar('total-loss', total_loss)

        return estimate_q_values, loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def predict(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_output, feed_dict=feed_dict)[0]

    def train(self, session, data_dict):
        states, actions, rewards, next_states, terminals, learning_rate, logging, global_step = \
            self.get_params(data_dict)

        if global_step % self.target_update == 0:
            self.update_target_network(session)

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = session.run(self.tf_target_output, feed_dict=feed_dict)

        max_q_values_next = np.max(q_values_next, axis=1)
        targets = rewards + np.subtract(1., terminals) * self.gamma * max_q_values_next

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_targets: targets}

        if logging:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
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

    def get_network_variables(self):
        return self.tf_network_variables

    def update_target_network(self, session):
        pred_params = sorted(self.tf_network_variables, key=lambda v: v.name)
        target_params = sorted(self.tf_target_variables, key=lambda v: v.name)
        update_ops = []
        for p1, p2 in zip(pred_params, target_params):
            op = p2.assign(p1)
            update_ops.append(op)
        session.run(update_ops)
