from fruit.networks.config.base import A3CConfig, MapA3CConfig
from fruit.networks.config.base import DQNConfig
import tensorflow as tf
from functools import partial
from fruit.utils.image import *


# Configuration for Atari games using A3C
class JairA3CConfig(A3CConfig):

    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01,
                 global_norm_clipping=40, debug_mode=False, gamma = 0.99, stochastic=True, num_of_objs=1, weights=None,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):

        self.num_of_objs = num_of_objs
        self.weights = weights

        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, beta=beta, global_norm_clipping=global_norm_clipping,
                         debug_mode=debug_mode, gamma=gamma, stochastic=stochastic,
                         optimizer=optimizer)

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = []
        for i in range(self.num_of_objs):
            rewards.append(self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards_" + str(i)))
        return inputs, actions, rewards, inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')
        layers = []
        for i in range(self.num_of_objs):
            layers.append(self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3_' + str(i)))
        actor_output = self.layer_manager.create_output(layer_3, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = []
        for i in range(self.num_of_objs):
            critic_output.append(tf.reshape(self.layer_manager.create_output(layers[i], 1, activation_fn='linear', scope='tf_critic_output_' + str(i)), [-1]))
        return actor_output, critic_output

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = []
            scale_critic_diff = []
            for i in range(self.num_of_objs):
                c = tf.subtract(self.tf_critic_output[i], self.tf_rewards[i])
                critic_diff.append(c)
                scale_critic_diff.append(c * self.weights[i])
            critic_total_diff = scale_critic_diff[0]
            for i in range(self.num_of_objs - 1):
                critic_total_diff = critic_total_diff + scale_critic_diff[i+1]

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_total_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            critic_loss = []
            for i in range(self.num_of_objs):
                critic_loss.append(tf.nn.l2_loss(critic_diff[i]) * 0.5)
                tf.summary.scalar('critic-loss-' + str(i), critic_loss[i])

        with tf.name_scope('total-loss'):
            critic_total_loss = critic_loss[0]
            for i in range(self.num_of_objs - 1):
                critic_total_loss = critic_total_loss + critic_loss[i+1]
            total_loss = tf.reduce_sum(critic_total_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        curr_reward = [0] * self.num_of_objs
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1])
            curr_reward = np.max(q_values, axis=1)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = np.add(reward, np.multiply(self.gamma, curr_reward))
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        reward_inputs = []
        for i in range(self.num_of_objs):
            reward_list = []
            for j in range(len(td_rewards)):
                reward_list.append(td_rewards[j][i])
            reward_inputs.append(reward_list)

        # print(reward_inputs)

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions,
                     self.tf_learning_rate: learning_rate}
        for i in range(self.num_of_objs):
            feed_dict.update({self.tf_rewards[i]:reward_inputs[i]})

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        critic_output = []
        for i in range(self.num_of_objs):
            critic_output.append(session.run(self.tf_critic_output[i], feed_dict=feed_dict))
        return critic_output

    def get_network_variables(self):
        return self.tf_network_variables


# Configuration for Atari games using A3C
class AtariA3C2Config(A3CConfig):
    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_actions_2, self.tf_rewards, self.tf_rewards_2, self.tf_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_actor_output_2, self.tf_critic_output_2 = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss, self.tf_total_loss_2 = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        with tf.name_scope('shared-optimizer-2'):
            self.tf_train_step_2, self.tf_learning_rate_2 = self.create_train_step_2()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        actions_2 = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions_2")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")
        rewards_2 = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards_2")
        return inputs, actions, actions_2, rewards, rewards_2, inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')
        actor_output = self.layer_manager.create_output(layer_3, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_3, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])

        layer_3_2 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3_2')
        actor_output_2 = self.layer_manager.create_output(layer_3_2, self.output_size, activation_fn='softmax', scope='tf_actor_output_2')
        critic_output_2 = self.layer_manager.create_output(layer_3_2, 1, activation_fn='linear', scope='tf_critic_output_2')
        critic_output_2 = tf.reshape(critic_output_2, [-1])

        return actor_output, critic_output, actor_output_2, critic_output_2

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_train_step_2(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate_2")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads-2'):
            gradients = optimizer.compute_gradients(self.tf_total_loss_2)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate-2', learning_rate)
        return train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        with tf.name_scope('critic-reward-diff-2'):
            critic_diff_2 = tf.subtract(self.tf_critic_output_2, self.tf_rewards_2)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('log-of-actor-policy-2'):
            actions_one_hot_2 = tf.one_hot(self.tf_actions_2, depth=self.output_size, name='one-hot-2',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy_2 = tf.log(self.tf_actor_output_2 + 1e-6)
            log_policy_one_hot_2 = tf.multiply(log_policy_2, actions_one_hot_2)
            log_policy_action_2 = tf.reduce_sum(log_policy_one_hot_2, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-entropy-2'):
            actor_entropy_2 = tf.reduce_sum(tf.multiply(self.tf_actor_output_2, log_policy_2))

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('actor-loss-2'):
            actor_loss_2 = tf.reduce_sum(tf.multiply(log_policy_action_2, tf.stop_gradient(critic_diff_2)))

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('critic-loss-2'):
            critic_loss_2 = tf.nn.l2_loss(critic_diff_2) * 0.5

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        with tf.name_scope('total-loss-2'):
            total_loss_2 = tf.reduce_sum(critic_loss_2 + actor_loss_2 + (actor_entropy_2 * self.beta))

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss, total_loss_2

    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def get_prediction_2(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_actor_output_2, feed_dict=feed_dict)[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values_2(session, next_states[-1])
            curr_reward = max(q_values)

        td_rewards_2 = []
        for reward in reversed(other_data[1]):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards_2.append(curr_reward)

        td_rewards_2 = list(reversed(td_rewards_2))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_actions_2: other_data[0],
                     self.tf_rewards: td_rewards, self.tf_rewards_2: td_rewards_2,
                     self.tf_learning_rate: learning_rate, self.tf_learning_rate_2: learning_rate}

        session.run([self.tf_train_step_2], feed_dict=feed_dict)

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_q_values_2(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_critic_output_2, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


# Configuration for Atari games using A3C
class AtariA3CMAConfig(A3CConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01, global_norm_clipping=40,
                 debug_mode=False, gamma=0.99, stochastic=True,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):
        super().__init__(environment, initial_learning_rate, history_length, beta, global_norm_clipping, debug_mode,
                         gamma, stochastic, optimizer)

        self.num_of_agents = environment.get_num_of_agents()

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        print("Input shape:", self.input_shape)
        print("Output size:", self.output_size)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions_list, self.tf_rewards, self.tf_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actors_list, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        actions_list = []
        for i in range(self.num_of_agents):
            actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions_" + str(i))
            actions_list.append(actions)

        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        return inputs, actions_list, rewards, inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2')

        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu',
                                                                  scope='tf_layer_3')

        critic_output = self.layer_manager.create_output(layer_3, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])

        actors_list = []
        for i in range(self.num_of_agents):
            actors_list.append(self.layer_manager.create_output(layer_3, self.output_size, activation_fn='softmax',
                                                                scope='tf_actor_output_' + str(i)))

        return actors_list, critic_output

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        actor_losses = None
        for i in range(self.num_of_agents):
            with tf.name_scope('log-of-actor-policy'):
                actions_one_hot = tf.one_hot(self.tf_actions_list[i], depth=self.output_size, name='one-hot-' + str(i),
                                             on_value=1.0, off_value=0.0, dtype=tf.float32)
                log_policy = tf.log(self.tf_actors_list[i] + 1e-6)
                log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actors_list[i], log_policy))
                tf.summary.scalar("actor-entropy-" + str(i), actor_entropy)

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
                tf.summary.scalar('actor-loss-' + str(i), actor_loss)

            with tf.name_scope('actor-losses'):
                if actor_losses is None:
                    actor_losses = actor_loss + (actor_entropy * self.beta)
                else:
                    actor_losses = actor_losses + actor_loss + (actor_entropy * self.beta)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_losses)
            tf.summary.scalar('total-loss', total_loss)

        return total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_actors_list, feed_dict=feed_dict)

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        actions_ = []
        for i in range(self.num_of_agents):
            acs = []
            for j in range(len(actions)):
                acs.append(actions[j][i])
            actions_.append(acs)

        actions = actions_

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states,
                     self.tf_rewards: td_rewards,
                     self.tf_learning_rate: learning_rate}

        for i in range(self.num_of_agents):
            feed_dict[self.tf_actions_list[i]] = actions[i]

        session.run([self.tf_train_step], feed_dict=feed_dict)

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                total_loss = session.run(self.tf_total_loss, feed_dict=feed_dict)
                print("TOTAL LOSS")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


class AtariA3CMapMAConfig(MapA3CConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01, global_norm_clipping=40,
                 debug_mode=False, gamma=0.99, stochastic=True, use_map=True, alpha=1,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):
        super().__init__(environment, initial_learning_rate, history_length, beta, global_norm_clipping, debug_mode,
                         gamma, stochastic, optimizer)

        self.num_of_agents = environment.get_num_of_agents()
        self.use_map = use_map
        self.alpha = alpha

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        print("Input shape:", self.input_shape)
        print("Output size:", self.output_size)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions_list, self.tf_rewards, self.tf_inputs_norm,\
                self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actors_list, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_total_loss, self.tf_actions_one_hot_list, self.tf_log_policy_list, self.tf_log_policy_one_hot_list, \
            self.tf_log_policy_action_list, self.tf_actor_entropy_list, self.tf_actor_loss_list, \
            self.tf_actor_losses, self.tf_critic_loss, self.tf_critic_diff = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        actions_list = []
        for i in range(self.num_of_agents):
            actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions_" + str(i))
            actions_list.append(actions)

        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        return inputs, actions_list, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):

        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2')

        layer_1_map = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1_map')
        layer_2_map = self.layer_manager.create_conv_layer(layer_1_map, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2_map')

        layer_cat = tf.concat([layer_2, layer_2_map], axis=3)

        if self.use_map:
            layer_3 = self.layer_manager.create_fully_connected_layer(layer_cat, 256, activation_fn='relu',
                                                                      scope='tf_layer_3')
        else:
            layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu',
                                                                      scope='tf_layer_3')

        layer_4 = self.layer_manager.create_fully_connected_layer(layer_3, 256, activation_fn='relu',
                                                                  scope='tf_layer_4')

        critic_output = self.layer_manager.create_output(layer_4, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])

        actors_list = []
        for i in range(self.num_of_agents):
            actors_list.append(self.layer_manager.create_output(layer_4, self.output_size, activation_fn='softmax', scope='tf_actor_output_' + str(i)))

        return actors_list, critic_output

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        actor_losses = None
        actions_one_hot_list = []
        log_policy_list = []
        log_policy_one_hot_list = []
        log_policy_action_list = []
        actor_entropy_list = []
        actor_loss_list = []

        for i in range(self.num_of_agents):
            with tf.name_scope('log-of-actor-policy'):
                actions_one_hot = tf.one_hot(self.tf_actions_list[i], depth=self.output_size, name='one-hot-' + str(i),
                                             on_value=1.0, off_value=0.0, dtype=tf.float32)
                log_policy = tf.log(self.tf_actors_list[i] + 1e-6)
                log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

                actions_one_hot_list.append(actions_one_hot)
                log_policy_list.append(log_policy)
                log_policy_one_hot_list.append(log_policy_one_hot)
                log_policy_action_list.append(log_policy_action)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actors_list[i], log_policy))
                tf.summary.scalar("actor-entropy-" + str(i), actor_entropy)

                actor_entropy_list.append(actor_entropy)

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
                tf.summary.scalar('actor-loss-' + str(i), actor_loss)

                actor_loss_list.append(actor_loss)

            with tf.name_scope('actor-losses'):
                if actor_losses is None:
                    actor_losses = actor_loss + (actor_entropy * self.beta)
                else:
                    actor_losses = actor_losses + actor_loss + (actor_entropy * self.beta)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5 * self.alpha
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_losses)
            tf.summary.scalar('total-loss', total_loss)

        return total_loss, actions_one_hot_list, log_policy_list, log_policy_one_hot_list, \
               log_policy_action_list, actor_entropy_list, actor_loss_list, actor_losses, critic_loss, critic_diff


    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state, map):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map}
        return session.run(self.tf_actors_list, feed_dict=feed_dict)

    def train_network(self, session, states, actions, rewards, next_states, terminals, maps, learning_rate, summaries,
                      global_step, other_data):

        actions_ = []
        for i in range(self.num_of_agents):
            acs = []
            for j in range(len(actions)):
                acs.append(actions[j][i])
            actions_.append(acs)

        actions = actions_

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], maps[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states,
                     self.tf_map_inputs: maps,
                     self.tf_rewards: td_rewards,
                     self.tf_learning_rate: learning_rate}

        for i in range(self.num_of_agents):
            feed_dict[self.tf_actions_list[i]] = actions[i]

        session.run([self.tf_train_step], feed_dict=feed_dict)

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)

                print("REWARDS:")
                print(rewards)

                print("TD REWARDS")
                print(td_rewards)

                print("MAP")
                print(maps)

                print("LEARNING RATE:", learning_rate)

                debug_val = [self.tf_total_loss, self.tf_actions_one_hot_list, self.tf_log_policy_list,
                             self.tf_log_policy_one_hot_list, self.tf_log_policy_action_list,
                             self.tf_actor_entropy_list, self.tf_actor_loss_list,
                             self.tf_actor_losses, self.tf_critic_loss, self.tf_critic_diff,
                             self.tf_critic_output, self.tf_actors_list]

                total_loss, actions_one_hot_list, log_policy_list, log_policy_one_hot_list, \
                log_policy_action_list, actor_entropy_list, actor_loss_list, actor_losses, \
                critic_loss, critic_diff, critic_output, actors_list = \
                    session.run(debug_val, feed_dict=feed_dict)

                print("TOTAL LOSS")
                print(total_loss)

                print('ACTION ONE HOT LIST')
                print(actions_one_hot_list)

                print('LOG POLICY LIST')
                print(log_policy_list)

                print('LOG POLICY ONE HOT')
                print(log_policy_one_hot_list)

                print('LOG POLICY ACTION LIST')
                print(log_policy_action_list)

                print('ACTOR ENTROPY')
                print(actor_entropy_list)

                print('ACTOR LOSS LIST')
                print(actor_loss_list)

                print('ACTOR LOSSES')
                print(actor_losses)

                print('CRITIC LOSS')
                print(critic_loss)

                print('CRITIC DIFF')
                print(critic_diff)

                print('CRITIC OUTPUT')
                print(critic_output)

                print('ACTORS LIST')
                print(actors_list)

                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, map):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [map]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


class AtariA3CTestMapMAConfig(MapA3CConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01, global_norm_clipping=40,
                 debug_mode=False, gamma=0.99, stochastic=True, use_map=True, alpha=1,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):
        super().__init__(environment, initial_learning_rate, history_length, beta, global_norm_clipping, debug_mode,
                         gamma, stochastic, optimizer)

        self.num_of_agents = environment.get_num_of_agents()
        self.use_map = use_map
        self.alpha = alpha

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        print("Input shape:", self.input_shape)
        print("Output size:", self.output_size)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions_list, self.tf_rewards, self.tf_inputs_norm,\
                self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actors_list, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_total_loss, self.tf_actions_one_hot_list, self.tf_log_policy_list, self.tf_log_policy_one_hot_list, \
            self.tf_log_policy_action_list, self.tf_actor_entropy_list, self.tf_actor_loss_list, \
            self.tf_actor_losses, self.tf_critic_loss, self.tf_critic_diff = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        actions_list = []
        for i in range(self.num_of_agents):
            actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions_" + str(i))
            actions_list.append(actions)

        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        return inputs, actions_list, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):

        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 64, activation_fn='relu', scope='tf_layer_3')


        layer_1_map = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 8, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1_map')
        layer_2_map = self.layer_manager.create_conv_layer(layer_1_map, 16, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2_map')
        layer_3_map = self.layer_manager.create_fully_connected_layer(layer_2_map, 64, activation_fn='relu',
                                                                  scope='tf_layer_3_map')

        layer_cat = tf.concat([layer_3, layer_3_map], axis=1)

        layer_4 = self.layer_manager.create_fully_connected_layer(layer_cat, 128, activation_fn='relu',
                                                                      scope='tf_layer_4')

        critic_output = self.layer_manager.create_output(layer_4, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])

        actors_list = []
        for i in range(self.num_of_agents):
            layer_5 = self.layer_manager.create_fully_connected_layer(layer_cat, 128, activation_fn='relu',
                                                                      scope='tf_layer_5_' + str(i))
            actors_list.append(self.layer_manager.create_output(layer_5, self.output_size, activation_fn='softmax',
                                                                scope='tf_actor_output_' + str(i)))

        return actors_list, critic_output

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        actor_losses = None
        actions_one_hot_list = []
        log_policy_list = []
        log_policy_one_hot_list = []
        log_policy_action_list = []
        actor_entropy_list = []
        actor_loss_list = []

        for i in range(self.num_of_agents):
            with tf.name_scope('log-of-actor-policy'):
                actions_one_hot = tf.one_hot(self.tf_actions_list[i], depth=self.output_size, name='one-hot-' + str(i),
                                             on_value=1.0, off_value=0.0, dtype=tf.float32)
                log_policy = tf.log(self.tf_actors_list[i] + 1e-6)
                log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

                actions_one_hot_list.append(actions_one_hot)
                log_policy_list.append(log_policy)
                log_policy_one_hot_list.append(log_policy_one_hot)
                log_policy_action_list.append(log_policy_action)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actors_list[i], log_policy))
                tf.summary.scalar("actor-entropy-" + str(i), actor_entropy)

                actor_entropy_list.append(actor_entropy)

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
                tf.summary.scalar('actor-loss-' + str(i), actor_loss)

                actor_loss_list.append(actor_loss)

            with tf.name_scope('actor-losses'):
                if actor_losses is None:
                    actor_losses = actor_loss + (actor_entropy * self.beta)
                else:
                    actor_losses = actor_losses + actor_loss + (actor_entropy * self.beta)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5 * self.alpha
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_losses)
            tf.summary.scalar('total-loss', total_loss)

        return total_loss, actions_one_hot_list, log_policy_list, log_policy_one_hot_list, \
               log_policy_action_list, actor_entropy_list, actor_loss_list, actor_losses, critic_loss, critic_diff


    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state, map):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map}
        return session.run(self.tf_actors_list, feed_dict=feed_dict)

    def train_network(self, session, states, actions, rewards, next_states, terminals, maps, learning_rate, summaries,
                      global_step, other_data):

        actions_ = []
        for i in range(self.num_of_agents):
            acs = []
            for j in range(len(actions)):
                acs.append(actions[j][i])
            actions_.append(acs)

        actions = actions_

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], maps[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states,
                     self.tf_map_inputs: maps,
                     self.tf_rewards: td_rewards,
                     self.tf_learning_rate: learning_rate}

        for i in range(self.num_of_agents):
            feed_dict[self.tf_actions_list[i]] = actions[i]

        session.run([self.tf_train_step], feed_dict=feed_dict)

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)

                print("REWARDS:")
                print(rewards)

                print("TD REWARDS")
                print(td_rewards)

                print("MAP")
                print(maps)

                print("LEARNING RATE:", learning_rate)

                debug_val = [self.tf_total_loss, self.tf_actions_one_hot_list, self.tf_log_policy_list,
                             self.tf_log_policy_one_hot_list, self.tf_log_policy_action_list,
                             self.tf_actor_entropy_list, self.tf_actor_loss_list,
                             self.tf_actor_losses, self.tf_critic_loss, self.tf_critic_diff,
                             self.tf_critic_output, self.tf_actors_list]

                total_loss, actions_one_hot_list, log_policy_list, log_policy_one_hot_list, \
                log_policy_action_list, actor_entropy_list, actor_loss_list, actor_losses, \
                critic_loss, critic_diff, critic_output, actors_list = \
                    session.run(debug_val, feed_dict=feed_dict)

                print("TOTAL LOSS")
                print(total_loss)

                print('ACTION ONE HOT LIST')
                print(actions_one_hot_list)

                print('LOG POLICY LIST')
                print(log_policy_list)

                print('LOG POLICY ONE HOT')
                print(log_policy_one_hot_list)

                print('LOG POLICY ACTION LIST')
                print(log_policy_action_list)

                print('ACTOR ENTROPY')
                print(actor_entropy_list)

                print('ACTOR LOSS LIST')
                print(actor_loss_list)

                print('ACTOR LOSSES')
                print(actor_losses)

                print('CRITIC LOSS')
                print(critic_loss)

                print('CRITIC DIFF')
                print(critic_diff)

                print('CRITIC OUTPUT')
                print(critic_output)

                print('ACTORS LIST')
                print(actors_list)

                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, map):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [map]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


class AtariA3CFakeMapMAConfig(MapA3CConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01, global_norm_clipping=40,
                 debug_mode=False, gamma=0.99, stochastic=True,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):
        super().__init__(environment, initial_learning_rate, history_length, beta, global_norm_clipping, debug_mode,
                         gamma, stochastic, optimizer)

        self.num_of_agents = environment.get_num_of_agents()

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)

        print("Input shape:", self.input_shape)
        print("Output size:", self.output_size)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions_list, self.tf_rewards, self.tf_inputs_norm,\
                self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actors_list, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        actions_list = []
        for i in range(self.num_of_agents):
            actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions_" + str(i))
            actions_list.append(actions)

        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        return inputs, actions_list, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):

        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2')

        layer_1_map = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 16, 8, strides=4, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_1_map')
        layer_2_map = self.layer_manager.create_conv_layer(layer_1_map, 32, 4, strides=2, activation_fn='relu',
                                                       padding='valid', scope='tf_layer_2_map')

        # layer_cat = tf.concat([layer_2, layer_2_map], axis=3)

        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu',
                                                                  scope='tf_layer_3')
        layer_4 = self.layer_manager.create_fully_connected_layer(layer_3, 256, activation_fn='relu',
                                                                  scope='tf_layer_4')

        critic_output = self.layer_manager.create_output(layer_4, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])

        actors_list = []
        for i in range(self.num_of_agents):
            actors_list.append(self.layer_manager.create_output(layer_4, self.output_size, activation_fn='softmax', scope='tf_actor_output_' + str(i)))

        return actors_list, critic_output

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        actor_losses = None
        for i in range(self.num_of_agents):
            with tf.name_scope('log-of-actor-policy'):
                actions_one_hot = tf.one_hot(self.tf_actions_list[i], depth=self.output_size, name='one-hot-' + str(i),
                                             on_value=1.0, off_value=0.0, dtype=tf.float32)
                log_policy = tf.log(self.tf_actors_list[i] + 1e-6)
                log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
                log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

            with tf.name_scope('actor-entropy'):
                actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actors_list[i], log_policy))
                tf.summary.scalar("actor-entropy-" + str(i), actor_entropy)

            with tf.name_scope('actor-loss'):
                actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
                tf.summary.scalar('actor-loss-' + str(i), actor_loss)

            with tf.name_scope('actor-losses'):
                if actor_losses is None:
                    actor_losses = actor_loss + (actor_entropy * self.beta)
                else:
                    actor_losses = actor_losses + actor_loss + (actor_entropy * self.beta)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_losses)
            tf.summary.scalar('total-loss', total_loss)

        return total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state, map):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map}
        return session.run(self.tf_actors_list, feed_dict=feed_dict)

    def train_network(self, session, states, actions, rewards, next_states, terminals, maps, learning_rate, summaries,
                      global_step, other_data):

        actions_ = []
        for i in range(self.num_of_agents):
            acs = []
            for j in range(len(actions)):
                acs.append(actions[j][i])
            actions_.append(acs)

        actions = actions_

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], maps[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states,
                     self.tf_map_inputs: maps,
                     self.tf_rewards: td_rewards,
                     self.tf_learning_rate: learning_rate}

        for i in range(self.num_of_agents):
            feed_dict[self.tf_actions_list[i]] = actions[i]

        session.run([self.tf_train_step], feed_dict=feed_dict)

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                total_loss = session.run(self.tf_total_loss, feed_dict=feed_dict)
                print("TOTAL LOSS")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, map):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [map]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


class Jair2A3CConfig(A3CConfig):

    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01,
                 global_norm_clipping=40, debug_mode=False, gamma = 0.99, stochastic=True, num_of_objs=1, weights=None,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):

        self.num_of_objs = num_of_objs
        self.weights = weights

        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, beta=beta, global_norm_clipping=global_norm_clipping,
                         debug_mode=debug_mode, gamma=gamma, stochastic=stochastic,
                         optimizer=optimizer)

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)
        self.weight_input_shape = [self.num_of_objs]

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm, self.tf_weights_input = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.l1, self.l2, self.l3 = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")
        weights = self.layer_manager.create_input(tf.float32, shape=[None] + self.weight_input_shape, name="tf_weights")
        return inputs, actions, rewards, inputs_norm, weights

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')
        layer_4 = self.layer_manager.create_fully_connected_layer(self.tf_weights_input, 256, activation_fn='relu', scope='tf_layer_4')

        layer_5 = tf.concat([layer_3, layer_4], axis=1)

        actor_output = self.layer_manager.create_output(layer_5, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_5, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        return actor_output, critic_output, layer_3, layer_4, layer_5

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def get_prediction_with_weight(self, session, state, thread_id):
        feed_dict = {self.tf_inputs: state, self.tf_weights_input: [self.weights[thread_id]]}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        thread_id = other_data
        # print(thread_id, self.weights[thread_id])
        temp = []
        batch_size = np.array(rewards).shape[0]
        for i in range(batch_size):
            temp.append(np.multiply(rewards[i], self.weights[thread_id]))
        rewards = np.sum(temp, axis=1)

        weights_data = []
        for i in range(batch_size):
            weights_data.append(self.weights[thread_id])

        # print(weights_data)

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values_with_weight(session, next_states[-1], thread_id)
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_weights_input: weights_data,
                     self.tf_learning_rate: learning_rate}

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                 self.tf_log_policy, self.tf_log_policy_action,
                                 self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                 self.tf_total_loss,
                                 ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_q_values_with_weight(self, session, state, thread_id):
        feed_dict = {self.tf_inputs: [state], self.tf_weights_input: [self.weights[thread_id]]}
        output = session.run(self.tf_critic_output, feed_dict=feed_dict)
        return output

    def get_network_variables(self):
        return self.tf_network_variables


class NIPSA3CConfig(A3CConfig):

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.get_shape())
        self.output_size = len(self.action_range)
        self.map_shape = [self.history_length] + list(self.state_space.get_shape())

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm, \
            self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_combined_layer, self.layer_3, self.layer_6 = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.map_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        return inputs, actions, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        # layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')

        layer_4 = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_4')
        layer_5 = self.layer_manager.create_conv_layer(layer_4, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_5')
        # layer_6 = self.layer_manager.create_fully_connected_layer(layer_5, 256, activation_fn='relu', scope='tf_layer_6')

        layer_7 = tf.concat([layer_2, layer_5], axis=3)
        layer_8 = self.layer_manager.create_fully_connected_layer(layer_7, 256, activation_fn='relu', scope='tf_layer_8')

        actor_output = self.layer_manager.create_output(layer_8, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_8, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        return actor_output, critic_output, layer_7, layer_2, layer_5

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction_with_map(self, session, state, map_data):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map_data}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], other_data[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_map_inputs: other_data,
                     self.tf_learning_rate: learning_rate}

        # print("LAYER_3", session.run(self.layer_3, feed_dict=feed_dict))
        # print("LAYER_6", session.run(self.layer_6, feed_dict=feed_dict))
        # print("LAYER_7", session.run(self.tf_combined_layer, feed_dict=feed_dict))

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, other_data):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [other_data]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


class MANIPSA3CConfig(A3CConfig):

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)
        self.map_shape = [self.history_length] + list(self.state_space.shape)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm, \
            self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_combined_layer, self.layer_3, self.layer_6 = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('network-2'):
            self.tf_actor_output_2, self.tf_critic_output_2, self.tf_combined_layer_2, self.layer_3_2, self.layer_6_2 = self.create_network()
            self.tf_network_variables_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network-2')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.map_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        return inputs, actions, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        # layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')

        layer_4 = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_4')
        layer_5 = self.layer_manager.create_conv_layer(layer_4, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_5')
        # layer_6 = self.layer_manager.create_fully_connected_layer(layer_5, 256, activation_fn='relu', scope='tf_layer_6')

        layer_7 = tf.concat([layer_2, layer_5], axis=3)
        layer_8 = self.layer_manager.create_fully_connected_layer(layer_7, 256, activation_fn='relu', scope='tf_layer_8')

        actor_output = self.layer_manager.create_output(layer_8, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_8, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        return actor_output, critic_output, layer_7, layer_2, layer_5

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction_with_map(self, session, state, map_data):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map_data}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def get_prediction_with_map_2(self, session, state, map_data):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map_data}
        return session.run(self.tf_actor_output_2, feed_dict=feed_dict)[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], other_data[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_map_inputs: other_data,
                     self.tf_learning_rate: learning_rate}

        # print("LAYER_3", session.run(self.layer_3, feed_dict=feed_dict))
        # print("LAYER_6", session.run(self.layer_6, feed_dict=feed_dict))
        # print("LAYER_7", session.run(self.tf_combined_layer, feed_dict=feed_dict))

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, other_data):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [other_data]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables

    def update_network_2(self, session):
        pred_params = self.tf_network_variables
        pred_params = sorted(pred_params, key=lambda v: v.name)
        target_params = self.tf_network_variables_2
        target_params = sorted(target_params, key=lambda v: v.name)
        ops = []
        for p1, p2 in zip(pred_params, target_params):
            op = p2.assign(p1)
            ops.append(op)
        session.run(ops)


class MSCSA3CConfig(A3CConfig):

    def init_config(self):

        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = 2
        self.map_shape = [self.history_length] + list(self.state_space.shape)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm, \
            self.tf_map_inputs, self.tf_map_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_combined_layer, self.layer_3, self.layer_6 = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")

        map_inputs = self.layer_manager.create_input(tf.uint8, [None] + self.map_shape, name="tf_map_inputs")
        map_inputs_norm = tf.cast(tf.transpose(map_inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0

        return inputs, actions, rewards, inputs_norm, map_inputs, map_inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        # layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')

        layer_4 = self.layer_manager.create_conv_layer(self.tf_map_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_4')
        layer_5 = self.layer_manager.create_conv_layer(layer_4, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_5')
        # layer_6 = self.layer_manager.create_fully_connected_layer(layer_5, 256, activation_fn='relu', scope='tf_layer_6')

        layer_7 = tf.concat([layer_2, layer_5], axis=3)
        layer_8 = self.layer_manager.create_fully_connected_layer(layer_7, 256, activation_fn='relu', scope='tf_layer_8')

        actor_output = self.layer_manager.create_output(layer_8, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_8, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        return actor_output, critic_output, layer_7, layer_2, layer_5

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            critic_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            critic_loss = tf.nn.l2_loss(critic_diff) * 0.5
            tf.summary.scalar('critic-loss', critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction_with_map(self, session, state, map_data):
        feed_dict = {self.tf_inputs: state, self.tf_map_inputs: map_data}
        return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], other_data[-1])
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_map_inputs: other_data,
                     self.tf_learning_rate: learning_rate}

        # print("LAYER_3", session.run(self.layer_3, feed_dict=feed_dict))
        # print("LAYER_6", session.run(self.layer_6, feed_dict=feed_dict))
        # print("LAYER_7", session.run(self.tf_combined_layer, feed_dict=feed_dict))

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, other_data):
        feed_dict = {self.tf_inputs: [state], self.tf_map_inputs: [other_data]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


# Divide and conquer paper (SMC 2018)
class AtariDQA3CConfig(AtariA3CConfig):

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm = self.create_input()

        with tf.name_scope('input_2'):
            self.tf_inputs_2, self.tf_actions_2, self.tf_rewards_2, self.tf_inputs_norm_2 = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('network_2'):
            self.tf_actor_output_2, self.tf_critic_output_2 = self.create_network()
            self.tf_network_variables_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network_2')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables, self.tf_network_variables_2

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state}
        probs_1 = session.run(self.tf_actor_output, feed_dict=feed_dict)[0]

        for i in range(len(state[0])):
            blacken_image(state[0][i], 84, 84, 0, 42, 0, 84)

        feed_dict_2 = {self.tf_inputs: state}
        probs_2 = session.run(self.tf_actor_output_2, feed_dict=feed_dict_2)[0]

        return probs_1, probs_2

    def update_dq_network(self, session):
        pred_params = self.tf_network_variables
        pred_params = sorted(pred_params, key=lambda v: v.name)
        target_params = self.tf_network_variables_2
        target_params = sorted(target_params, key=lambda v: v.name)
        ops = []
        for p1, p2 in zip(pred_params, target_params):
            op = p2.assign(p1)
            ops.append(op)
        session.run(ops)


class AtariA3CTimeBasedConfig(A3CConfig):

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_time_rewards, \
            self.tf_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_time_critic_output = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.variable_scope('eval'):
            self.tf_eval_output, _, _ = self.create_network()
            self.tf_eval_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def get_eval_network(self):
        return self.tf_inputs, self.tf_eval_output, self.tf_eval_variables

    def create_input(self):
        inputs = self.layer_manager.create_input(tf.uint8, [None] + self.input_shape, name="tf_inputs")
        inputs_norm = tf.cast(tf.transpose(inputs, perm=[0, 2, 3, 1]), tf.float32) / 255.0
        actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
        rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")
        time_rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_time_rewards")
        return inputs, actions, rewards, time_rewards, inputs_norm

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')
        actor_output = self.layer_manager.create_output(layer_3, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_3, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        time_critic_output = self.layer_manager.create_output(layer_3, 1, activation_fn='linear', scope='tf_time_critic_output')
        time_critic_output = tf.reshape(time_critic_output, [-1])
        return actor_output, critic_output, time_critic_output

    def create_train_step(self):
        learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
        optimizer = self.optimizer(learning_rate=learning_rate)

        with tf.name_scope('compute-clip-grads'):
            gradients = optimizer.compute_gradients(self.tf_total_loss)
            tensors = [tensor for gradient, tensor in gradients]
            grads = [gradient for gradient, tensor in gradients]
            clipped_gradients, _ = tf.clip_by_global_norm(grads, self.global_norm_clipping)
            clipped_grads_tensors = zip(clipped_gradients, tensors)
            train_step = optimizer.apply_gradients(clipped_grads_tensors)
        tf.summary.scalar('learning-rate', learning_rate)
        summaries = tf.summary.merge_all()
        return summaries, train_step, learning_rate

    def create_loss(self):
        with tf.name_scope('critic-reward-diff'):
            #c_diff = tf.subtract(self.tf_critic_output, self.tf_rewards)
            time_critic_diff = tf.subtract(self.tf_time_critic_output, self.tf_time_rewards)
            #critic_diff = tf.add(c_diff, time_critic_diff)

        with tf.name_scope('log-of-actor-policy'):
            actions_one_hot = tf.one_hot(self.tf_actions, depth=self.output_size, name='one-hot',
                                         on_value=1.0, off_value=0.0, dtype=tf.float32)
            log_policy = tf.log(self.tf_actor_output + 1e-6)
            log_policy_one_hot = tf.multiply(log_policy, actions_one_hot)
            log_policy_action = tf.reduce_sum(log_policy_one_hot, axis=1)

        with tf.name_scope('actor-entropy'):
            actor_entropy = tf.reduce_sum(tf.multiply(self.tf_actor_output, log_policy))
            tf.summary.scalar("actor-entropy", actor_entropy)

        with tf.name_scope('actor-loss'):
            actor_loss = tf.reduce_sum(tf.multiply(log_policy_action, tf.stop_gradient(time_critic_diff)))
            tf.summary.scalar('actor-loss', actor_loss)

        with tf.name_scope('critic-loss'):
            #critic_loss = tf.nn.l2_loss(c_diff) * 0.5
            time_critic_loss = tf.nn.l2_loss(time_critic_diff) * 0.5
            tf.summary.scalar('critic-loss', time_critic_loss)

        with tf.name_scope('total-loss'):
            total_loss = tf.reduce_sum(time_critic_loss + actor_loss + (actor_entropy * self.beta))
            tf.summary.scalar('total-loss', total_loss)

        return time_critic_diff, log_policy, log_policy_action, actor_entropy, actor_loss, time_critic_loss, total_loss

    def get_input_shape(self):
        return self.input_shape

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state}
        ret = session.run(self.tf_actor_output, feed_dict=feed_dict)[0]
        return ret

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, other_data):

        time_rewards = []
        for i in range(len(rewards)):
            if rewards[i] == 0:
                time_rewards.append(-1)
            else:
                time_rewards.append(0)

        curr_reward = 0
        time_curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1])
            curr_reward = max(q_values)
            time_curr_reward = max(self.get_time_q_values(session, next_states[-1]))

        td_rewards = []
        time_td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        for reward in reversed(time_rewards):
            time_curr_reward = reward + self.gamma * time_curr_reward
            time_td_rewards.append(time_curr_reward)

        time_td_rewards = list(reversed(time_td_rewards))
        td_rewards = list(reversed(td_rewards))

        # print(td_rewards)
        print(time_rewards)
        print(time_td_rewards)

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_time_rewards: time_td_rewards,
                     self.tf_learning_rate: learning_rate}

        summaries = True
        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)

    def get_time_q_values(self, session, state):
        feed_dict = {self.tf_inputs: [state]}
        return session.run(self.tf_time_critic_output, feed_dict=feed_dict)

    def get_network_variables(self):
        return self.tf_network_variables


class AtariA3CLSTMConfig(AtariA3CConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01,
                 global_norm_clipping=40, debug_mode=False, gamma = 0.99, stochastic=True, lstm_size=256,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):

        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, beta=beta, global_norm_clipping=global_norm_clipping,
                         debug_mode=debug_mode, gamma=gamma, stochastic=stochastic,
                         optimizer=optimizer)

        self.lstm_size = lstm_size
        self.prev_lstm_state = None
        self.reset_config()

    def init_config(self):
        self.input_shape = [self.history_length] + list(self.state_space.shape)
        self.output_size = len(self.action_range)

        with tf.name_scope('input'):
            self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_inputs_norm = self.create_input()

        with tf.variable_scope('network'):
            self.tf_actor_output, self.tf_critic_output, self.tf_init_state, self.tf_new_state = self.create_network()
            self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')

        with tf.name_scope('loss'):
            self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
            self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()

        with tf.name_scope('shared-optimizer'):
            self.tf_summaries, self.tf_train_step, self.tf_learning_rate = self.create_train_step()

        return self.tf_network_variables

    def create_network(self):
        layer_1 = self.layer_manager.create_conv_layer(self.tf_inputs_norm, 16, 8, strides=4, activation_fn='relu', padding='valid', scope='tf_layer_1')
        layer_2 = self.layer_manager.create_conv_layer(layer_1, 32, 4, strides=2, activation_fn='relu', padding='valid', scope='tf_layer_2')
        layer_3 = self.layer_manager.create_fully_connected_layer(layer_2, 256, activation_fn='relu', scope='tf_layer_3')
        layer_4, init_state, new_state = self.layer_manager.create_basic_lstm_layer(layer_3, 256, self.lstm_size, scope='tf_layer_4')
        actor_output = self.layer_manager.create_output(layer_4, self.output_size, activation_fn='softmax', scope='tf_actor_output')
        critic_output = self.layer_manager.create_output(layer_4, 1, activation_fn='linear', scope='tf_critic_output')
        critic_output = tf.reshape(critic_output, [-1])
        return actor_output, critic_output, init_state, new_state

    def reset_config(self):
        self.prev_lstm_state = (np.zeros((1, self.lstm_size)), np.zeros((1, self.lstm_size)))

    def get_lstm_state(self):
        return self.prev_lstm_state

    def get_prediction(self, session, state):
        feed_dict = {self.tf_inputs: state, self.tf_init_state: self.prev_lstm_state}
        output, new_state = session.run([self.tf_actor_output, self.tf_new_state], feed_dict=feed_dict)
        self.prev_lstm_state = new_state
        return output[0]

    def train_network(self, session, states, actions, rewards, next_states, terminals, learning_rate, summaries,
                      global_step, lstm_state):

        curr_reward = 0
        if not terminals[-1]:
            q_values = self.get_q_values(session, next_states[-1], lstm_state)
            curr_reward = max(q_values)

        td_rewards = []
        for reward in reversed(rewards):
            curr_reward = reward + self.gamma * curr_reward
            td_rewards.append(curr_reward)

        td_rewards = list(reversed(td_rewards))

        feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: td_rewards,
                     self.tf_learning_rate: learning_rate, self.tf_init_state: lstm_state}

        if summaries:
            if self.debug_mode:
                print("##############################################################################")
                print("ACTIONS:")
                print(actions)
                print("REWARDS:")
                print(rewards)
                print("LEARNING RATE:", learning_rate)
                actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
                    session.run([self.tf_actor_output, self.tf_critic_output,
                                         self.tf_log_policy, self.tf_log_policy_action,
                                         self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
                                         self.tf_total_loss,
                                         ], feed_dict=feed_dict)
                print("ACTOR OUTPUT:")
                print(actor_output)
                print("CRITIC OUTPUT:")
                print(critic_output)
                print("LOG POLICY:")
                print(log_policy)
                print("LOG POLICY ACTION:")
                print(log_policy_action)
                print("CRITIC DIFF:")
                print(critic_diff)
                print("CRITIC LOSS:")
                print(critic_loss)
                print("ACTOR LOSS:")
                print(actor_loss)
                print("ACTOR ENTROPY:")
                print(actor_entropy)
                print("TOTAL LOSS:")
                print(total_loss)
                print("##############################################################################")
            return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
        else:
            return session.run([self.tf_train_step], feed_dict=feed_dict)

    def get_q_values(self, session, state, lstm_state=None):
        feed_dict = {self.tf_inputs: [state], self.tf_init_state: lstm_state}
        return session.run(self.tf_critic_output, feed_dict=feed_dict)





# Configuration for CartPole
# class CartPoleA3CConfig(AtariA3CConfig):
#     def __init__(self, environment, initial_learning_rate=0.004, history_length=1, beta=0.01,
#                  global_norm_clipping=40, debug_mode=False,
#                  optimizer=tf.train.RMSPropOptimizer(learning_rate=0.004, decay=0.99, epsilon=0.1)):
#         super().__init__(environment=environment, initial_learning_rate=initial_learning_rate, history_length=history_length,
#                          beta=beta, global_norm_clipping=global_norm_clipping, debug_mode=debug_mode,
#                          optimizer=optimizer)
#         self.prev_actor_state = None
#         self.prev_critic_state = None
#
#     def init_config(self):
#         self.input_shape = [self.history_length] + list(self.state_space.shape)
#         self.output_size = len(self.action_range)
#
#         with tf.name_scope('input'):
#             self.tf_inputs, self.tf_actions, self.tf_rewards, self.tf_learning_rate = self.create_input()
#
#         with tf.variable_scope('network'):
#             self.tf_actor_output, self.tf_critic_output, self.tf_init_actor_state, self.tf_new_actor_state, \
#                 self.tf_init_critic_state, self.tf_new_critic_state = self.create_network()
#
#             self.tf_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network')
#
#         with tf.name_scope('loss'):
#             self.tf_critic_diff, self.tf_log_policy, self.tf_log_policy_action,\
#             self.tf_actor_entropy, self.tf_actor_loss, self.tf_critic_loss, self.tf_total_loss = self.create_loss()
#
#         with tf.name_scope('optimizer'):
#             self.tf_summaries, self.tf_train_step = self.create_train_step()
#
#     def create_input(self):
#         inputs = self.layer_manager.create_input(tf.float32, [None] + self.get_input_shape(), name="tf_inputs")
#         actions = self.layer_manager.create_input(tf.int32, shape=[None], name="tf_actions")
#         rewards = self.layer_manager.create_input(tf.float32, shape=[None], name="tf_rewards")
#         learning_rate = self.layer_manager.create_input(tf.float32, shape=None, name="tf_learning_rate")
#         return inputs, actions, rewards, learning_rate
#
#     def create_network(self):
#         # Actor
#         layer_1 = self.layer_manager.create_fully_connected_layer(self.tf_inputs, 200, activation_fn='relu', scope='tf_layer_1')
#         layer_2, initial_lstm, new_lstm = self.layer_manager.create_basic_lstm_layer(layer_1, 128, scope='tf_lstm_output')
#         actor_output = self.layer_manager.create_output(layer_2, 2, activation_fn='softmax', scope='tf_actor_output')
#
#         # Critic
#         layer_1_1 = self.layer_manager.create_fully_connected_layer(self.tf_inputs, 200, activation_fn='relu', scope='tf_layer_1_1')
#         layer_2_1, initial_lstm_1, new_lstm_1 = self.layer_manager.create_basic_lstm_layer(layer_1_1, 128, scope='tf_lstm_output_1')
#         critic_output = self.layer_manager.create_output(layer_2_1, 1, activation_fn='linear', scope='tf_critic_output')
#         critic_output = tf.reshape(critic_output, [-1])
#
#         return actor_output, critic_output, initial_lstm, new_lstm, initial_lstm_1, new_lstm_1
#
#     def get_prediction(self, session, state):
#         feed_dict = {self.tf_inputs: [state]}
#         return session.run(self.tf_actor_output, feed_dict=feed_dict)[0]
#
#     def train_network(self, session, states, actions, rewards, learning_rate, summaries):
#         feed_dict = {self.tf_inputs: states, self.tf_actions: actions, self.tf_rewards: rewards,
#                      self.tf_learning_rate: learning_rate}
#
#         if summaries:
#             if self.debug_mode:
#                 print("################################################################")
#                 print("STATES:")
#                 print(states)
#                 print("ACTIONS:")
#                 print(actions)
#                 print("REWARDS:")
#                 print(rewards)
#                 print("LEARNING RATE:", learning_rate)
#                 actor_output, critic_output, log_policy, log_policy_action, critic_diff, critic_loss, actor_loss, actor_entropy, total_loss = \
#                     session.run([self.tf_actor_output, self.tf_critic_output,
#                                          self.tf_log_policy, self.tf_log_policy_action,
#                                          self.tf_critic_diff, self.tf_critic_loss, self.tf_actor_loss, self.tf_actor_entropy,
#                                          self.tf_total_loss,
#                                          ], feed_dict=feed_dict)
#                 print("ACTOR OUTPUT:")
#                 print(actor_output)
#                 print("CRITIC OUTPUT:")
#                 print(critic_output)
#                 print("LOG POLICY:")
#                 print(log_policy)
#                 print("LOG POLICY ACTION:")
#                 print(log_policy_action)
#                 print("CRITIC DIFF:")
#                 print(critic_diff)
#                 print("CRITIC LOSS:")
#                 print(critic_loss)
#                 print("ACTOR LOSS:")
#                 print(actor_loss)
#                 print("ACTOR ENTROPY:")
#                 print(actor_entropy)
#                 print("TOTAL LOSS:")
#                 print(total_loss)
#                 print("################################################################")
#             return session.run([self.tf_summaries, self.tf_train_step], feed_dict=feed_dict)[0]
#         else:
#             return session.run([self.tf_train_step], feed_dict=feed_dict)
#
#     def get_q_values(self, session, state):
#         feed_dict = {self.tf_inputs: [state]}
#         return session.run(self.tf_critic_output, feed_dict=feed_dict)
#
#     def reset_config(self):
#         self.prev_actor_state = (np.zeros((1, 128)), np.zeros((1, 128)))
#         self.prev_critic_state = (np.zeros((1, 128)), np.zeros((1, 128)))
