from fruit.networks.config.base import Config
import numpy as np
import tensorflow as tf
from functools import partial


class MODQNConfig(Config):
    def __init__(self, environment, initial_learning_rate=0.0001, debug_mode=False, gamma = 0.9,
                 target_network_update=1000, num_of_objectives=2, linear_thresholds=None,
                 TLO_thresholds=None, is_linear=False, using_cnn=False, history_length=1,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=1e-6)):
        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, debug_mode=debug_mode, gamma=gamma,
                         stochastic_policy=False, optimizer=optimizer)
        self.target_update = target_network_update
        self.num_of_objs = num_of_objectives
        self.thresholds = linear_thresholds
        self.TLO_thresholds = TLO_thresholds
        self.using_cnn = using_cnn
        self.is_linear = is_linear
        if self.thresholds is None:
            self.thresholds = [1/self.num_of_objs] * (self.num_of_objs)
        if self.TLO_thresholds is None:
            self.TLO_thresholds = [0.] * (self.num_of_objs-1)
        self.optimizer = self.optimizer(learning_rate=initial_learning_rate)

    def get_num_of_objectives(self):
        return self.num_of_objs

    def init_config(self):
        self.input_shape = list(self.state_space.shape)
        print(self.input_shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_targets = self.__create_input()

        with tf.variable_scope('network'):
            self.tf_output = self.__create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('target'):
            self.tf_target_output = self.__create_network()
            self.tf_target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        with tf.name_scope('loss'):
            self.tf_est_q_values, self.tf_loss, self.tf_total_loss = self.__create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step = self.__create_train_step()

        return self.tf_network_variables

    def __create_input(self):
        inputs = self.layer_manager.create_input(tf.float32, [None] + self.input_shape, name="tf_inputs")
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        targets = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_targets")
        return inputs, actions, targets

    def __create_network(self):
        layer_1 = self.layer_manager.create_fully_connected_layer(self.tf_inputs, 512, activation_fn='relu',
                                                                  scope='tf_layer_1')
        layer_2 = self.layer_manager.create_fully_connected_layer(layer_1, 512, activation_fn='relu',
                                                                  scope='tf_layer_2')
        output = self.layer_manager.create_output(layer_2, self.output_size, scope='tf_output')
        return output

    def __create_train_step(self):
        train_step = self.optimizer.minimize(self.tf_total_loss)
        summaries = tf.summary.merge_all()
        return summaries, train_step

    def __create_loss(self):

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

    def get_prediction(self, session, state):
        if isinstance(state, list):
            feed_dict = {self.tf_inputs: [state]}
        else:
            feed_dict = {self.tf_inputs: [[state]]}
        result = session.run(self.tf_output, feed_dict=feed_dict)[0]
        return result

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries, global_step, other):

        if global_step % self.target_update == 0:
            if self.debug_mode:
                print("Target update !")
            self.update_target_network(session)

        batch_size = np.array(states).shape[0]

        temp = []
        if self.is_linear:
            for i in range(batch_size):
                temp.append(np.multiply(rewards[i], self.thresholds))
            rewards = np.sum(temp, axis=1)
        else:
            # Use TLO thresholds
            for i in range(batch_size):
                for j in range(self.num_of_objs-1):
                    if rewards[i][j] > self.TLO_thresholds[j]:
                        rewards[i][j] = self.TLO_thresholds[j]
            for i in range(batch_size):
                temp.append(np.multiply(rewards[i], self.thresholds))
            rewards = np.sum(temp, axis=1)

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = session.run(self.tf_target_output, feed_dict=feed_dict)
        max_q_values_next = np.max(q_values_next, axis=1)
        targets = rewards + np.subtract(1., terminals) * self.gamma * max_q_values_next

        #print(targets)
        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_targets: targets}

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("STATES:")
                print(states)
                print("NEXT STATES:")
                print(next_states)
                print(batch_size, q_values_next, targets)
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                estimate_q_values, loss, total_loss = \
                    session.run([self.tf_est_q_values, self.tf_loss, self.tf_total_loss], feed_dict=feed_dict)
                print("VALUES:")
                print(estimate_q_values)
                print("LOSS:")
                print(loss)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")

            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]

        else:
            return session.run(self.tf_train_step, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables

    def update_target_network(self, session):
        pred_params = self.tf_network_variables
        pred_params = sorted(pred_params, key=lambda v: v.name)
        target_params = self.tf_target_variables
        target_params = sorted(target_params, key=lambda v: v.name)
        ops = []
        for p1, p2 in zip(pred_params, target_params):
            op = p2.assign(p1)
            ops.append(op)
        session.run(ops)


class MOExDQNConfig(MODQNConfig):

    def init_config(self):
        self.input_shape = list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_targets, self.tf_inputs_norm = self.__create_input()

        with tf.variable_scope('network'):
            self.tf_output = self.__create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('target'):
            self.tf_target_output = self.__create_network()
            self.tf_target_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        with tf.name_scope('loss'):
            self.tf_est_q_values, self.tf_loss, self.tf_total_loss = self.__create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step = self.__create_train_step()

        return self.tf_network_variables

    def __create_input(self):
        if self.using_cnn:
            inputs = self.layer_manager.create_input(tf.uint8, [None] + [self.history_length] + self.input_shape,
                                                     name="tf_inputs")
            inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        else:
            inputs = self.layer_manager.create_input(tf.float32, [None] + self.input_shape, name="tf_inputs")
            inputs_norm = None
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        targets = []
        for i in range(self.num_of_objs):
            targets.append(self.layer_manager.create_input(tf.float32, shape=[None], name="tf_targets" + str(i)))
        return inputs, actions, targets, inputs_norm

    def __create_network(self):
        if self.using_cnn:
            layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 32, 8, strides=4, activation_fn='relu',
                                                           padding='valid', scope='tf_layer_1')
            layer_2 = self.layer_manager.create_conv_layer(layer_1, 64, 4, strides=2, activation_fn='relu', padding='valid',
                                                           scope='tf_layer_2')
            layer_3 = self.layer_manager.create_conv_layer(layer_2, 64, 3, strides=1, activation_fn='relu', padding='valid',
                                                           scope='tf_layer_3')
            layer_4 = self.layer_manager.create_fully_connected_layer(layer_3, 512, activation_fn='relu',
                                                                      scope='tf_layer_4')
        else:
            layer_1 = self.layer_manager.create_fully_connected_layer(self.tf_inputs, 512, activation_fn='relu',
                                                                      scope='tf_layer_1')
            layer_4 = self.layer_manager.create_fully_connected_layer(layer_1, 512, activation_fn='relu',
                                                                      scope='tf_layer_2')
        output = []
        for i in range(self.num_of_objs):
            output.append(self.layer_manager.create_output(layer_4, self.output_size, scope='tf_output' + str(i)))
        return output

    def __create_train_step(self):
        train_step = self.optimizer.minimize(self.tf_total_loss)
        summaries = tf.summary.merge_all()
        return summaries, train_step

    def __create_loss(self):

        with tf.name_scope('estimate_q_values'):
            estimate_q_values = []
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            for i in range(self.num_of_objs):
                estimate_q_values.append(tf.reduce_sum(tf.multiply(self.tf_output[i], actions_one_hot), axis=1))

        with tf.name_scope('loss'):
            loss = []
            for i in range(self.num_of_objs):
                loss.append(tf.squared_difference(estimate_q_values, self.tf_targets))

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_mean(loss[0])
            for i in range(self.num_of_objs):
                if i > 0:
                    total_loss = tf.reduce_sum(tf.reduce_mean(loss[i]))
            tf.summary.scalar('total-loss', total_loss)

        return estimate_q_values, loss, total_loss

    def get_input_shape(self):
        if self.using_cnn:
            return [self.history_length] + self.input_shape
        else:
            return self.input_shape

    def get_prediction(self, session, state):
        if self.using_cnn:
            feed_dict = {self.tf_inputs: state}
        else:
            if isinstance(state, list):
                feed_dict = {self.tf_inputs: [state]}
            else:
                feed_dict = {self.tf_inputs: [[state]]}
        result = 0.
        if self.is_linear:
            for i in range(self.num_of_objs):
                result = result + session.run(self.tf_output[i], feed_dict=feed_dict)[0]*self.thresholds[i]
        else:
            for i in range(self.num_of_objs):
               pred = session.run(self.tf_output[i], feed_dict=feed_dict)[0]
               if i < self.num_of_objs-1:
                   for j in range(len(pred)):
                       if pred[j] > self.TLO_thresholds[i]:
                           pred[j] = self.TLO_thresholds[i]
               result = result + pred * self.thresholds[i]
        return result

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries, global_step, other):

        if global_step % self.target_update == 0:
            if self.debug_mode:
                print("Target update !")
            self.update_target_network(session)

        batch_size = np.array(states).shape[0]

        r = [[0. for _ in range(len(rewards))] for _ in range(self.num_of_objs)]

        for i in range(self.num_of_objs):
            for j in range(len(rewards)):
                r[i][j] = rewards[j][i]

        feed_dict = {self.tf_inputs: next_states}
        q_values_next = []
        org_values = []
        max_q_values_next = []
        targets = []
        result = 0.
        if self.is_linear:
            for i in range(self.num_of_objs):
                q_values_next.append(session.run(self.tf_target_output[i], feed_dict=feed_dict))
                max_q_values_next.append(np.max(q_values_next[i], axis=1))
                targets.append(r[i] + np.subtract(1., terminals) * self.gamma * max_q_values_next[i])
        else:
            for i in range(self.num_of_objs):
                q_values_next.append(session.run(self.tf_target_output[i], feed_dict=feed_dict))
                org_values.append(q_values_next[i])
                if i < self.num_of_objs-1:
                    for j in range(batch_size):
                        for k in range(self.output_size):
                            if q_values_next[i][j][k] > self.TLO_thresholds[i]:
                                q_values_next[i][j][k] = self.TLO_thresholds[i]
                result = result + q_values_next[i] * self.thresholds[i]
            # if global_step % 1000 == 0:
            #    print(q_values_next)
            greedy_actions = np.argmax(result, axis=1)
            max_q_values_next = [[] for _ in range(self.num_of_objs)]
            for i in range(self.num_of_objs):
                for j in range(batch_size):
                    max_q_values_next[i].append(org_values[i][j][greedy_actions[j]])
                targets.append(r[i] + np.subtract(1., terminals) * self.gamma * max_q_values_next[i])

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions}
        for i in range(self.num_of_objs):
            feed_dict.update({self.tf_targets[i]:targets[i]})

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("STATES:")
                print(states)
                print("NEXT STATES:")
                print(next_states)
                print(batch_size, q_values_next, targets)
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                total_loss = session.run(self.tf_total_loss, feed_dict=feed_dict)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")

            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]

        else:
            return session.run(self.tf_train_step, feed_dict=feed_dict)


